import Header from "./components/Header";
import UploadForm from "./components/UploadForm";

function App() {
  return (
    <div className="min-h-screen flex flex-col bg-black text-white font-sans">
      <Header />

      <main className="flex flex-col items-center justify-center px-4 py-10 gap-16 flex-grow">
        <section className="text-center max-w-2xl">
          <h1 className="text-4xl font-semibold mb-4">Финансовые прогнозы</h1>
          <p className="text-zinc-400 mb-8">
            Загрузите CSV-файл с отчётностью компании и получите DOCX-отчёт с прогнозами, построенными на основе моделей <strong>XGBoost</strong> и <strong>LSTM</strong>.
          </p>
          <UploadForm />
        </section>

        <section className="max-w-2xl text-center text-zinc-400">
          <h2 className="text-2xl font-medium mb-3">О технологии</h2>
          <p>
            Модель <strong>XGBoost</strong> делает точечные прогнозы, а <strong>LSTM</strong> анализирует временные ряды.
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
