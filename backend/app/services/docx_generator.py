from docx import Document
from docx.shared import Pt
import logging
import numpy as np
import matplotlib.pyplot as plt
import tempfile


def build_report(df, preds, report_path):
    logger = logging.getLogger("docx_generator")
    logger.info("Building report", extra={"path": report_path})

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

    doc = Document()
    doc.add_heading('Отчёт о финансовом прогнозе', level=1)

    doc.add_paragraph(
        'Спасибо за то, что выбрали наш продукт. В этом отчёте представлены прогнозы ключевых финансовых показателей на следующий период и их интерпретация с учётом ваших потребностей как инвестора/владельца бизнеса.'
    )

    doc.add_heading('Расшифровка показателей', level=2)
    for col, desc in definitions.items():
        p = doc.add_paragraph(style='List Number')
        run = p.add_run(f'{labels[col]} ({col}): ')
        run.bold = True
        p.add_run(desc)

    doc.add_heading('Как работают наши модели', level=2)
    doc.add_paragraph(
        '• XGBoost — это проверенный алгоритм, строящий сильные прогнозы на основе набора решающих деревьев. '  
        'Он быстро обрабатывает данные и выдаёт точечные оценки будущих значений.'
    )
    doc.add_paragraph(
        '• LSTM — это разновидность нейросети, учитывающая последовательность исторических данных. '  
        'Она выявляет скрытые закономерности и тренды, чтобы дать более глубокий анализ временных зависимостей.'
    )

    doc.add_heading('Исходные данные от вас', level=2)
    table_in = doc.add_table(rows=1, cols=len(df.columns))
    table_in.style = 'Table Grid'
    hdr = table_in.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr[i].text = labels.get(col, col)
    for _, row in df.iterrows():
        cells = table_in.add_row().cells
        for i, col in enumerate(df.columns):
            cells[i].text = f'{row[col]:,}'

    xgb_preds = np.array(preds.get('xgboost', []), dtype=float)
    lstm_preds = np.array(preds.get('lstm', []), dtype=float)

    doc.add_heading('Прогнозы по моделям', level=2)

    doc.add_heading('XGBoost', level=3)
    tbl_xgb = doc.add_table(rows=1, cols=2)
    tbl_xgb.style = 'Table Grid'
    hdr = tbl_xgb.rows[0].cells
    hdr[0].text, hdr[1].text = '№', 'Значение'
    for i, val in enumerate(xgb_preds, 1):
        c = tbl_xgb.add_row().cells
        c[0].text, c[1].text = str(i), f'{val:.2f}'

    doc.add_heading('LSTM', level=3)
    tbl_lstm = doc.add_table(rows=1, cols=2)
    tbl_lstm.style = 'Table Grid'
    hdr = tbl_lstm.rows[0].cells
    hdr[0].text, hdr[1].text = '№', 'Значение'
    for i, val in enumerate(lstm_preds, 1):
        c = tbl_lstm.add_row().cells
        c[0].text, c[1].text = str(i), f'{val:.2f}'


    doc.add_heading('Ключевые метрики прогнозов', level=2)
    def add_metrics(name, arr):
        doc.add_heading(name, level=3)
        metrics = {
            'Среднее': arr.mean(),
            'Медиана': np.median(arr),
            'Стандартное отклонение': arr.std(),
            'Минимум': arr.min(),
            'Максимум': arr.max()
        }
        tbl = doc.add_table(rows=len(metrics), cols=2)
        tbl.style = 'Table Grid'
        for idx, (mname, mval) in enumerate(metrics.items()):
            tbl.rows[idx].cells[0].text = mname
            tbl.rows[idx].cells[1].text = f'{mval:.2f}'
        return metrics

    xgb_metrics = add_metrics('XGBoost', xgb_preds)
    lstm_metrics = add_metrics('LSTM', lstm_preds)


    doc.add_heading('Что означают метрики', level=2)
    doc.add_paragraph(
        '• Среднее показывает «центральный» прогноз.'
        '• Медиана — устойчивое значение, не чувствительное к выбросам.'
        '• Стандартное отклонение демонстрирует разброс прогнозов.'
        '• Минимум/Максимум — крайние значения.')


    doc.add_heading('Графики для визуализации', level=2)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(xgb_preds)+1), xgb_preds, label='XGBoost')
        ax.plot(range(1, len(lstm_preds)+1), lstm_preds, '--', label='LSTM')
        ax.set_title('Сравнение прогнозов')
        ax.set_xlabel('Период')
        ax.set_ylabel('Значение')
        ax.legend()
        fig.savefig(tmp.name, bbox_inches='tight')
        plt.close(fig)
        doc.add_picture(tmp.name)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].hist(xgb_preds, bins=5)
        axes[0].set_title('Распределение XGBoost')
        axes[1].hist(lstm_preds, bins=5)
        axes[1].set_title('Распределение LSTM')
        fig.tight_layout()
        fig.savefig(tmp.name, bbox_inches='tight')
        plt.close(fig)
        doc.add_picture(tmp.name)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig, ax = plt.subplots()
        labels_metrics = list(xgb_metrics.keys())
        x_vals = np.arange(len(labels_metrics))
        ax.bar(x_vals - 0.2, list(xgb_metrics.values()), width=0.4, label='XGBoost')
        ax.bar(x_vals + 0.2, list(lstm_metrics.values()), width=0.4, label='LSTM')
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels_metrics, rotation=45)
        ax.set_title('Сравнение метрик')
        ax.legend()
        fig.savefig(tmp.name, bbox_inches='tight')
        plt.close(fig)
        doc.add_picture(tmp.name)


    doc.add_heading('Рекомендации', level=2)
    doc.add_paragraph(
        '• Для быстрого получения точечных оценок используйте XGBoost.'
        '• Для анализа трендов и сезонности обращайтесь к LSTM.'
        '• Сочетайте оба прогноза при принятии решений — это поможет получить всесторонний взгляд на будущее.')


    doc.save(report_path)
    logger.info("Report saved successfully", extra={"path": report_path})
