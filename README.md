# 0. (Необязательно) «Начисто» удалить старые контейнеры, образы и кэш
#    Делайте это, если предыдущие сборки/запуски оставили мусор
#    ВАЖНО: -a --volumes стирает анонимные тома; ваши каталоги ./data, ./models и др. на хосте не тронутся

docker compose down --volumes --remove-orphans   # остановить стек и удалить его контейнеры
docker builder prune --all                       # очистить кэш BuildKit (слои сборки)
docker image prune -a                            # удалить все неиспользуемые образы
# docker system prune -a --volumes               # 💣 максимально жёсткая чистка (осторожно!)

```bash
# 1. Clone / copy project files
cd financial-report-generator

# 2. Build services (first time)
docker compose build

# 3. Download dataset slice (requires HF_TOKEN in .env)
docker compose run --rm backend python backend/training/download_rfsd.py

# 4. Train models ( ~10 min on CPU )
docker compose run --rm backend python backend/training/train_xgb.py
docker compose run --rm backend python backend/training/train_lstm.py

# 5. Launch the full stack
docker compose up -d
# Frontend: http://localhost:5173  |  API docs: http://localhost:8000/docs