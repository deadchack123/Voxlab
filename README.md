# SST-TTS

Веб-приложение для транскрибации речи и синтеза голоса. Работает локально в Docker.

**Транскрибация (STT):** Whisper (faster-whisper) — модели от tiny до large-v3-turbo, 13+ языков, VAD-фильтрация, диаризация (разделение по спикерам), экспорт в TXT/SRT/VTT/JSON.

**Синтез речи (TTS):** XTTS v2 — клонирование голоса по 30-секундному образцу, 17 языков, настройка температуры/скорости/тембра/высоты, пресеты голосов.

**Очистка текста:** LLM-очистка транскрибации (удаление слов-паразитов, исправление пунктуации). Поддержка Groq, OpenRouter, Anthropic.

## Быстрый старт

### 1. Скачать docker-compose.yml

```bash
mkdir sst-tts && cd sst-tts
curl -O https://raw.githubusercontent.com/deadchack123/sst-tts/main/docker-compose.yml
```

### 2. Запустить

```bash
docker compose up -d
```

Откройте http://localhost:5050

Готово! При первом запуске скачаются модели (~3 ГБ для Whisper large-v3-turbo). Модели кэшируются в Docker volumes и не скачиваются повторно.

## Настройка

Создайте файл `.env` рядом с `docker-compose.yml`:

```env
# Модель Whisper (по умолчанию large-v3-turbo)
# Варианты: tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo
WHISPER_MODEL=large-v3-turbo

# Папка для данных (по умолчанию ./data)
# DATA_DIR=./data
```

### LLM-очистка текста

Для очистки транскрибации нужен API-ключ одного из провайдеров. Настройка через веб-интерфейс (иконка шестерёнки):

| Провайдер | Модель | Получить ключ |
|-----------|--------|---------------|
| Groq | Llama 3.3 70B | https://console.groq.com |
| OpenRouter | Llama 3.3 70B (free) | https://openrouter.ai/keys |
| Anthropic | Claude Haiku | https://console.anthropic.com |

## Возможности

### Транскрибация
- Загрузка аудио/видео любого формата (mp3, wav, mp4, webm и др.)
- 8 моделей Whisper на выбор
- Автоопределение языка или ручной выбор (13 языков)
- VAD-фильтрация (отсечение тишины)
- Диаризация — определение кто говорит, переименование спикеров
- Пословный таймкод (word timestamps)
- Экспорт: TXT, SRT, VTT, JSON
- LLM-очистка результата

### Синтез речи (TTS)
- Клонирование голоса по образцу (загрузка файла или из транскрибации)
- 17 языков
- Пресеты голосов — сохранение и переиспользование
- Параметры: температура, скорость, top_k, top_p, repetition penalty
- Тюнинг голоса: интенсивность стиля, тембр, вариативность, сдвиг высоты

## Сборка из исходников

```bash
git clone https://github.com/deadchack123/sst-tts.git
cd sst-tts
docker compose up -d --build
```

## Системные требования

- Docker и Docker Compose
- 4+ ГБ RAM (8 ГБ рекомендуется для large моделей)
- ~5 ГБ диска (образ + модели)
- CPU (GPU не требуется, используется int8 квантизация)
