export default function ReportStatus({ state }) {
  if (state === "processing")
    return <p className="mt-2 text-sm text-zinc-400 animate-pulse">Генерация отчёта…</p>;
  if (state === "error")
    return <p className="mt-2 text-sm text-red-500">Что-то пошло не так.</p>;
  return null;
}
