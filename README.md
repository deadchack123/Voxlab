# Voxlab — Whisper + TTS Studio

Веб-приложение для транскрибации речи и синтеза голоса. Работает локально в Docker, не требует GPU.

## Быстрый старт

```bash
# 1. Скачать docker-compose.yml
mkdir voxlab && cd voxlab
curl -O https://raw.githubusercontent.com/deadchack123/voxlab/main/docker-compose.yml

# 2. Запустить
docker compose up -d
```

Открой http://localhost:5050 — готово!

При первом запуске скачается модель Whisper (~1.6 ГБ). Модель кэшируется и больше не качается.

## Как пользоваться

### Транскрибация (STT)

1. Открой http://localhost:5050
2. Загрузи источник:
   - **Файл** — перетащи аудио/видео в зону загрузки (или нажми для выбора)
   - **Ссылка** — вставь URL с YouTube, Rutube, VK и [1000+ сайтов](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
3. Жди результат — текст появится с таймкодами
4. Скачай результат в нужном формате: TXT, SRT, VTT или JSON

> **YouTube:** если видео не загружается — вставь cookies через кнопку "Вставить из буфера" (нужно расширение [Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) для Chrome)

**Настройки транскрибации** (раскрывающаяся панель):
- Модель Whisper — от tiny (быстро, неточно) до large-v3 (медленно, точно). По умолчанию large-v3-turbo — лучший баланс
- Язык — авто-определение или ручной выбор (13 языков)
- Диаризация — определяет кто говорит (спикер 1, спикер 2...), можно переименовать спикеров

**Очистка текста ИИ** (раскрывающаяся панель):
- Убирает слова-паразиты, исправляет пунктуацию
- Нужен API-ключ: Groq (бесплатно), OpenRouter (бесплатно) или Anthropic
- Настраивается прямо в интерфейсе

### Озвучка (TTS)

1. Переключись на вкладку **Озвучка (TTS)**
2. Загрузи образец голоса (аудио/видео, 10-30 сек чистой речи)
3. Подкрути голос если нужно — высота, скорость, выразительность, тембр
4. Введи текст и нажми **Озвучить**
5. Скачай результат (WAV)

**Источники голоса:**
- Загрузить файл — любое аудио/видео с голосом
- Из транскрибации — взять голос из ранее загруженного файла
- Записать с микрофона — записать голос прямо в браузере
- Из пресета — использовать ранее сохранённый голос

**Пресеты** — можно сохранить голос + настройки и переиспользовать.

## Настройка

Создай `.env` рядом с `docker-compose.yml`:

```env
# Модель Whisper (по умолчанию large-v3-turbo)
# Варианты: tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo
WHISPER_MODEL=large-v3-turbo
```

## Поддерживаемые форматы

- **Вход:** MP4, WebM, MP3, WAV, M4A, OGG, FLAC и другие (всё что понимает ffmpeg)
- **Экспорт транскрибации:** TXT, SRT (субтитры), VTT (субтитры), JSON
- **Экспорт озвучки:** WAV

## Сборка из исходников

Для разработки и тестирования:

```bash
git clone https://github.com/deadchack123/voxlab.git
cd voxlab
docker compose -f docker-compose.dev.yml up -d --build
```

## GPU-ускорение (NVIDIA)

GPU даёт 10-20x ускорение транскрибации. Нужна видеокарта NVIDIA с поддержкой CUDA.

### Запуск с GPU

```bash
docker compose -f docker-compose.gpu.yml up -d --build
```

В логах должно быть: `Using device: cuda (compute_type: float16)`

### Требования для GPU

- **Linux:** установить [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Windows (WSL2):**
  1. Установить последний [NVIDIA драйвер для Windows](https://www.nvidia.com/drivers/) (Game Ready или Studio)
  2. Включить WSL2 в Docker Desktop (Settings → General → Use WSL2)
  3. Docker Desktop автоматически подхватит GPU через WSL2
- **macOS:** не поддерживается (нет NVIDIA GPU)

### Проверка GPU в контейнере

```bash
docker compose -f docker-compose.gpu.yml exec whisper python -c "import torch; print(torch.cuda.is_available())"
```

## Системные требования

- Docker и Docker Compose
- 4+ ГБ RAM (8 ГБ рекомендуется)
- ~5 ГБ диска (образ + модели)
- CPU (GPU не требуется)
- Работает на Linux, macOS (Intel и Apple Silicon), Windows (через Docker Desktop)
