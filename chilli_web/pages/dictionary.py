"""Data Dictionary -- web equivalent of ``ui.DataDictionaryPage``."""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, callback, dcc, html

from chilli_desktop.data_loader import build_data_dictionary, data_dictionary_markdown
from chilli_web import components as C
from chilli_web import page_common as PG
from chilli_web import server_state

dash.register_page(
    __name__, path="/dictionary", name="Data Dictionary",
    description="Auto-generated from the workbook's sheets and columns", order=12,
)


def layout(**_kwargs):
    return html.Div(
        [
            html.Div(id="dd-meta-box"),
            html.Div(
                [
                    html.Button(
                        "Export data dictionary as Markdown", id="dd-export-btn",
                        className="primary-button", n_clicks=0, style={"width": "auto", "padding": "8px 16px"},
                    ),
                    dcc.Download(id="dd-export-download"),
                ],
                style={"marginBottom": "16px"},
            ),
            C.section_header("Field-level dictionary"),
            html.Div(id="dd-fields-table"),
            C.section_header("Dataset coverage"),
            html.Div(id="dd-coverage-table"),
            C.section_header("Units and conversions read from the workbook"),
            html.Div(id="dd-units-table"),
            html.Div(id="dd-unmapped"),
            html.Div(id="dd-quality-box"),
        ]
    )


@callback(
    Output("dd-meta-box", "children"),
    Output("dd-fields-table", "children"),
    Output("dd-coverage-table", "children"),
    Output("dd-units-table", "children"),
    Output("dd-unmapped", "children"),
    Output("dd-quality-box", "children"),
    Input("filters-store", "data"),
    Input("theme-store", "data"),
    Input("reload-tick", "data"),
)
def _render(filters_data, theme_name, _tick):
    service, filters, theme = PG.current(filters_data, theme_name)
    data = service.data

    meta_box = C.info_box(
        f"Workbook: {data.path}\n"
        f"Read at {data.loaded_at:%d %b %Y %H:%M:%S} in {data.load_seconds:.2f}s.\n"
        f"Worksheets in file: {len(data.raw_shapes)} · mapped to analyses: "
        f"{len(data.datasets)} · unmapped: {len(data.unmapped_sheets)}.",
        tone="info",
    )

    fields_table = C.dataframe_table(
        build_data_dictionary(data), theme, title="Data dictionary", source=data.path.name,
        notes=["Generated at runtime by inspecting the workbook. 'Populated' is the share of "
               "rows with a value; 'Role' is inferred from the column header."],
        id="dd-fields", page_size=25,
    )

    coverage_table = C.dataframe_table(service.coverage_table(), theme, title="Coverage by dataset", source=data.path.name, id="dd-coverage")

    unit_rows: list[dict] = []
    for dataset in data.datasets.values():
        for label, weight in (dataset.meta.get("bag_weights_kg") or {}).items():
            unit_rows.append({"Sheet": dataset.sheet_name, "Field": label, "Conversion": f"1 bag = {weight:g} kg", "Read from": "column header text"})
        for note_key in ("primary_unit", "secondary_unit", "unit_note", "unit", "price_unit"):
            note = dataset.meta.get(note_key)
            if note:
                unit_rows.append({"Sheet": dataset.sheet_name, "Field": note_key.replace("_", " "), "Conversion": str(note), "Read from": "sheet annotation"})
    if unit_rows:
        units_table = C.dataframe_table(
            pd.DataFrame(unit_rows), theme, title="Units and conversions", source=data.path.name,
            notes=["No conversion is assumed anywhere in this application. Where a sheet "
                   "states no unit, quantities are reported in the sheet's own terms and "
                   "comparisons are confined to that sheet."],
            id="dd-units",
        )
    else:
        units_table = C.chart_card("Units and conversions", None, source="No unit annotation was found in any sheet.")

    unmapped = html.Div()
    if data.unmapped_sheets:
        unmapped_frame = pd.DataFrame(
            [{"Sheet": name, "Rows": data.raw_shapes.get(name, (0, 0))[0], "Columns": data.raw_shapes.get(name, (0, 0))[1]} for name in data.unmapped_sheets]
        )
        unmapped = html.Div(
            [
                C.section_header("Unmapped worksheets"),
                C.dataframe_table(unmapped_frame, theme, title="Present in the file but unused", source=data.path.name, notes=["No analysis in this application reads these sheets."], id="dd-unmapped"),
            ]
        )

    quality = service.data_quality_notes()
    boxes = []
    if quality:
        boxes.append(html.Div([C.section_header("Data quality"), C.info_box_items("Coverage limitations detected on load:", quality, tone="warning")]))
    if data.warnings:
        boxes.append(C.info_box_items("Parse warnings:", data.warnings, tone="warning"))
    quality_box = html.Div(boxes)

    return meta_box, fields_table, coverage_table, units_table, unmapped, quality_box


@callback(
    Output("dd-export-download", "data"),
    Input("dd-export-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _export_markdown(_n_clicks):
    service = server_state.get_service()
    return dict(content=data_dictionary_markdown(service.data), filename="DATA_DICTIONARY.md")
