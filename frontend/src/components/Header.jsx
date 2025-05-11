import { useState } from "react";
import ProfileModal from "./ProfileModal";

export default function Header() {
  const [open, setOpen] = useState(false);

  return (
    <header className="w-full flex items-center justify-between p-4 bg-black text-white border-b border-zinc-800">
      <h1 className="text-xl font-medium tracking-tight">Financial Forecast</h1>
      <button
        onClick={() => setOpen(true)}
        className="rounded-full p-2 hover:bg-zinc-800 transition"
        aria-label="Открыть профиль"
      >
        <svg
          className="w-6 h-6 text-white"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.5 20.25a8.25 8.25 0 1115 0" />
        </svg>
      </button>
      <ProfileModal isOpen={open} onClose={() => setOpen(false)} />
    </header>
  );
}
