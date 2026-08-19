"""Thread-based background-callback manager.

Dash's built-in ``DiskcacheManager`` runs every background callback (the
forecast sweep, the automated-insights sweep, the market-integration
battery, and Executive Summary's embedded forecast/insights panels) in a
freshly spawned *subprocess*, which re-imports the entire pandas/numpy/
scipy/statsmodels/xgboost stack a second time on top of the already-running
server process. On a memory-constrained host (observed: Render's Starter
plan, 512MB) that reliably gets the job killed partway through a model fit
-- the job polls successfully for a while, then the container kills the
process and the client sees a bare 502.

The desktop app's equivalent long-running work (chilli_desktop/ui.py) runs
on an in-process ``QThreadPool`` instead -- no second copy of the library
stack, because the worker thread already shares the request-handling
process's memory. This manager mirrors that: identical diskcache-backed
result/progress/set-props storage to ``DiskcacheManager`` (its private
``_make_job_fn`` is reused unchanged, so job semantics are byte-for-byte the
same), but executed on a plain Python thread instead of a subprocess.

Trade-off worth knowing: CPU-bound Python (not numpy/scipy's C internals,
which release the GIL) running in the background thread can make other
concurrent requests briefly slower, since they share one process's GIL --
the same trade-off the desktop app's QThreadPool already accepts. Given the
alternative was jobs dying outright under the subprocess model, this is the
better default for a memory-limited deployment.
"""

from __future__ import annotations

import threading

from dash.background_callback.managers import BaseBackgroundCallbackManager
from dash.background_callback.managers.diskcache_manager import _make_job_fn


class ThreadedDiskcacheManager(BaseBackgroundCallbackManager):
    def __init__(self, cache=None, cache_by=None, expire=None):
        import diskcache

        if cache is None:
            self.handle = diskcache.Cache()
        else:
            if not isinstance(cache, (diskcache.Cache, diskcache.FanoutCache)):
                raise ValueError(
                    "First argument must be a diskcache.Cache "
                    "or diskcache.FanoutCache object"
                )
            self.handle = cache

        self.expire = expire
        self._threads: dict[int, threading.Thread] = {}
        self._next_id = 1
        self._registry_lock = threading.Lock()
        super().__init__(cache_by)

    def _register_thread(self, thread: threading.Thread) -> int:
        with self._registry_lock:
            job_id = self._next_id
            self._next_id += 1
            self._threads[job_id] = thread
        return job_id

    def terminate_job(self, job):
        # There's no cooperative-cancellation hook into the analytics
        # functions themselves, so this just stops tracking the thread --
        # it's left to finish naturally rather than risking a hard kill
        # mid-computation (unlike the subprocess model, which can just be
        # OS-killed safely).
        if job is None:
            return
        with self._registry_lock:
            self._threads.pop(int(job), None)

    def terminate_unhealthy_job(self, job):
        return not self.job_running(job)

    def job_running(self, job):
        if not job:
            return False
        with self._registry_lock:
            thread = self._threads.get(int(job))
        return thread is not None and thread.is_alive()

    def make_job_fn(self, fn, progress, key=None):
        return _make_job_fn(fn, self.handle, progress)

    def clear_cache_entry(self, key):
        self.handle.delete(key)

    def get_or_create_signing_secret(self, generate):
        # ``add`` only sets the value if the key is absent (atomic in
        # diskcache), so the first worker wins and the rest read it back.
        self.handle.add(self.SIGNING_SECRET_KEY, generate())
        return self.handle.get(self.SIGNING_SECRET_KEY)

    def call_job_fn(self, key, job_fn, args, context):
        thread = threading.Thread(
            target=job_fn,
            args=(key, self._make_progress_key(key), args, context),
            daemon=True,
        )
        job_id = self._register_thread(thread)
        thread.start()
        return job_id

    def get_progress(self, key):
        progress_key = self._make_progress_key(key)
        progress_data = self.handle.get(progress_key)
        if progress_data:
            self.handle.delete(progress_key)
        return progress_data

    def result_ready(self, key):
        return self.handle.get(key) is not None

    def get_result(self, key, job):
        result = self.handle.get(key, self.UNDEFINED)
        if result is self.UNDEFINED:
            return self.UNDEFINED

        if self.cache_by is None:
            self.clear_cache_entry(key)
        else:
            if self.expire:
                self.handle.touch(key, expire=self.expire)

        self.clear_cache_entry(self._make_progress_key(key))

        if job:
            self.terminate_job(job)
        return result

    def get_updated_props(self, key):
        set_props_key = self._make_set_props_key(key)
        result = self.handle.get(set_props_key, self.UNDEFINED)
        if result is self.UNDEFINED:
            return {}
        self.clear_cache_entry(set_props_key)
        return result
