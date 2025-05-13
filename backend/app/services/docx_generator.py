import os
import logging
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches

logger = logging.getLogger('docx_generator')

def create_report(df, preds, report_path):
    logger.info('Building report', extra={'path': str(report_path)})

    definitions = {
        'line_1600': 'Выручка (Revenue) — итоговые поступления от продаж.',
        'line_2110': 'Себестоимость (Cost of Goods Sold) — затраты на производство товаров.',
        'line_2120': 'Операционная прибыль (Operating Income) — прибыль до учёта финансовых расходов.',
        'line_2400': 'Денежный поток (Cash Flow) — чистый приток/отток денежных средств.'
    }

    labels = {
        'line_1600': 'Выручка',
        'line_2110': 'Себестоимость',
        'line_2120': 'Операционная прибыль',
        'line_2400': 'Денежный поток'
    }

    xgb_preds = np.array(preds.get('xgboost', []), dtype=float)
    lstm_preds = np.array(preds.get('lstm', []), dtype=float)

    doc = Document()
    doc.add_heading('Отчёт о финансовом прогнозе', level=1)
    doc.add_paragraph(
        'Спасибо за то, что выбрали наш продукт. В этом отчёте представлены прогнозы ключевых финансовых показателей на следующий период и их интерпретация с учётом ваших потребностей как инвестора/владельца бизнеса.'
    )

    doc.add_heading('Расшифровка показателей', level=2)
    for code, desc in definitions.items():
        doc.add_paragraph(desc, style='List Number')

    doc.add_heading('Исходные данные от вас', level=2)
    n_rows = len(df) + 1
    n_cols = len(labels)
    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.style = 'Table Grid'
    for j, col in enumerate(labels):
        tbl.cell(0, j).text = labels[col]
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, col in enumerate(labels):
            value = getattr(row, col)
            if isinstance(value, float):
                tbl.cell(i, j).text = f'{value:,.2f}'
            else:
                tbl.cell(i, j).text = str(value)

    doc.add_heading('Прогнозы по моделям', level=2)
    doc.add_heading('XGBoost', level=3)
    tbl_xgb = doc.add_table(rows=len(xgb_preds) + 1, cols=2)
    tbl_xgb.style = 'Table Grid'
    tbl_xgb.cell(0, 0).text = '№'
    tbl_xgb.cell(0, 1).text = 'Значение'
    for i, pred in enumerate(xgb_preds, start=1):
        tbl_xgb.cell(i, 0).text = str(i)
        tbl_xgb.cell(i, 1).text = f'{pred:.2f}'
    doc.add_heading('LSTM', level=3)
    tbl_lstm = doc.add_table(rows=len(lstm_preds) + 1, cols=2)
    tbl_lstm.style = 'Table Grid'
    tbl_lstm.cell(0, 0).text = '№'
    tbl_lstm.cell(0, 1).text = 'Значение'
    for i, pred in enumerate(lstm_preds, start=1):
        tbl_lstm.cell(i, 0).text = str(i)
        tbl_lstm.cell(i, 1).text = f'{pred:.2f}'

    doc.add_heading('Ключевые метрики прогнозов', level=2)
    def add_metrics_section(name, arr):
        doc.add_heading(name, level=3)
        if len(arr) == 0:
            return None
        metrics = {
            'Среднее': float(np.mean(arr)),
            'Медиана': float(np.median(arr)),
            'Стандартное отклонение': float(np.std(arr)),
            'Минимум': float(np.min(arr)),
            'Максимум': float(np.max(arr))
        }
        tbl_m = doc.add_table(rows=len(metrics), cols=2)
        tbl_m.style = 'Table Grid'
        for idx, (mname, mval) in enumerate(metrics.items()):
            tbl_m.cell(idx, 0).text = mname
            tbl_m.cell(idx, 1).text = f'{mval:.2f}'
        return metrics

    xgb_metrics = add_metrics_section('XGBoost', xgb_preds)
    lstm_metrics = add_metrics_section('LSTM', lstm_preds)

    doc.add_paragraph(f"Прогноз XGBoost в среднем составил {xgb_metrics['Среднее']:.2f} единиц, "
                      f"при медиане {xgb_metrics['Медиана']:.2f} и стандартном отклонении {xgb_metrics['Стандартное отклонение']:.2f}.",
                      style='Intense Quote')

    doc.add_paragraph(f"Прогноз LSTM в среднем составил {lstm_metrics['Среднее']:.2f} единиц, "
                      f"при медиане {lstm_metrics['Медиана']:.2f} и стандартном отклонении {lstm_metrics['Стандартное отклонение']:.2f}.",
                      style='Intense Quote')

    doc.add_heading('Что означают метрики', level=3)
    doc.add_paragraph('Среднее показывает «центральный» прогноз.', style='List Bullet')
    doc.add_paragraph('Медиана — устойчивое значение, не чувствительное к выбросам.', style='List Bullet')
    doc.add_paragraph('Стандартное отклонение демонстрирует разброс прогнозов.', style='List Bullet')
    doc.add_paragraph('Минимум/Максимум — крайние значения.', style='List Bullet')

    doc.add_heading('Графики для визуализации', level=2)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(xgb_preds) + 1), xgb_preds, label='XGBoost')
    ax.plot(range(1, len(lstm_preds) + 1), lstm_preds, '--', label='LSTM')
    ax.legend()
    ax.set_title('Прогнозы моделей')
    ax.set_xlabel('Период')
    ax.set_ylabel('Значение')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig.savefig(tmp.name)
        tmp_path = tmp.name
    plt.close(fig)
    doc.add_picture(tmp_path, width=Inches(6))

    doc.add_heading('Рекомендации', level=2)
    doc.add_paragraph('Для быстрого получения точечных оценок используйте XGBoost.', style='List Bullet')
    doc.add_paragraph('Для анализа трендов и сезонности обращайтесь к LSTM.', style='List Bullet')
    doc.add_paragraph('Сочетайте оба прогноза при принятии решений — это поможет получить всесторонний взгляд на будущее.', style='List Bullet')

    doc.save(report_path)
    logger.info('Report generation completed', extra={'path': str(report_path)})
