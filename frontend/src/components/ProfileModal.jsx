import { useEffect } from "react";

export default function ProfileModal({ isOpen, onClose }) {
  useEffect(() => {
    const handleEsc = (e) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", handleEsc);
    return () => document.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-zinc-900 text-white p-6 rounded-2xl w-80 shadow-lg relative">
        <button
          onClick={onClose}
          className="absolute top-3 right-3 text-zinc-400 hover:text-white text-xl"
        >
          ×
        </button>
        <h2 className="text-xl font-semibold mb-2">Личный кабинет</h2>
        <p className="text-sm text-zinc-300"> In progress.</p>
      </div>
    </div>
  );
}
