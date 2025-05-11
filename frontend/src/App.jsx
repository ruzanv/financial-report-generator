import Header from "./components/Header";
import UploadForm from "./components/UploadForm";

function App() {
  return (
    <div className="min-h-screen flex flex-col bg-black text-white font-sans">
      <Header />

      <main className="flex flex-col items-center justify-center px-4 py-10 gap-16 flex-grow">
        <section className="text-center max-w-2xl">
          <h1 className="text-4xl font-semibold mb-4">Financial Forecast</h1>
          <p className="text-zinc-400 mb-8">
            Загрузите CSV-файл с отчётностью компании и получите DOCX-отчёт с прогнозами,
            построенными на основе моделей <strong>XGBoost</strong> и <strong>LSTM</strong>.
          </p>
          <h2 className="text-2xl font-medium mb-3">Необходимые колонки в CSV</h2>
          <ul className="list-disc list-inside space-y-2 text-left">
            <li>
              <strong>line_1600</strong>: Выручка — общий доход от основной деятельности компании.
            </li>
            <li>
              <strong>line_2110</strong>: Себестоимость — затраты на производство товаров или услуг.
            </li>
            <li>
              <strong>line_2120</strong>: Операционная прибыль — прибыль от основной деятельности до учета финансовых и налоговых расходов.
            </li>
            <li>
              <strong>line_2400</strong>: Денежный поток — чистый приток/отток денежных средств за период.
            </li>
          </ul>
          <UploadForm />
        </section>

        <section className="max-w-2xl text-center text-zinc-400">
          <h2 className="text-2xl font-medium mb-3">О технологии</h2>
          <p>
            <strong>XGBoost</strong> делает точечные прогнозы, опираясь на отдельные ключевые метрики бизнеса.
            <br />
            <strong>LSTM</strong> анализирует временной ряд денежных потоков и
            строит скользящие прогнозы на несколько периодов вперед.
            <br />
            Данные не передаются третьим лицам — всё обрабатывается локально.
          </p>
        </section>
      </main>

      <footer className="text-center py-6 border-t border-zinc-800 text-zinc-500 text-sm">
        © 2025 Financial Forecast Generator. Автор:{" "}
        <a
          href="https://github.com/ruzanv"
          className="text-white hover:underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          @ruzanv
        </a>
      </footer>
    </div>
  );
}

export default App;
