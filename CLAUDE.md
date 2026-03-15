# Voxlab

## Проект
Веб-приложение для транскрибации (Whisper) и синтеза речи (XTTS v2) с LLM-очисткой текста.
Работает в Docker-контейнере.

## Структура
- `app.py` — Flask бэкенд (Whisper, TTS, LLM cleanup, diarization)
- `static/index.html` — единый SPA-файл (HTML + CSS + JS)
- `Dockerfile` + `docker-compose.yml` — сборка и запуск
- `transcribe.py` — CLI-утилита для транскрибации

## Сборка и запуск
```bash
docker compose up -d --build
```
Сервер на порту 5000.

## TODO

### Docker-образ на GitHub Container Registry
- Настроить GitHub Actions для автоматической сборки образа
- Публиковать образ в ghcr.io чтобы пользователи могли `docker pull` без сборки
- Написать README.md с инструкцией по быстрой установке:
  - `docker pull ghcr.io/gorbunov-a/whisper-tts:latest`
  - `docker compose up -d` (с примером docker-compose.yml)
  - Настройка LLM-ключей
  - Описание возможностей (транскрибация, TTS, очистка текста)

### Текущее состояние LLM-очистки
- Очистка работает как единый текст (не посегментная)
- Поддержка провайдеров: Groq, OpenRouter, Anthropic
- Чанки по ~3000 символов с паузой 2 сек между запросами (rate limit)
