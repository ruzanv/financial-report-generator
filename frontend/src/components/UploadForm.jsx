import { useState } from "react";
import ReportStatus from "./ReportStatus";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState(null);

  const pollStatus = async (id) => {
    const res = await fetch(`/status/${id}`);
    const data = await res.json();
    if (data.state === "SUCCESS") {
      setStatus("done");
      window.location.href = data.download_url;
    } else if (data.state === "FAILURE") {
      setStatus("error");
    } else {
      setTimeout(() => pollStatus(id), 10000);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setTaskId(data.task_id);
    setStatus("processing");
    pollStatus(data.task_id);
  };

  return (
    <div className="p-4 max-w-xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          className="file-input file-input-bordered w-full"
        />
        <button type="submit" className="btn btn-primary w-full" disabled={!file}>
          Upload &amp; Generate Report
        </button>
      </form>

      <ReportStatus state={status} />
    </div>
  );
}
