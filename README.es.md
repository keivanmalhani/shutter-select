# shutter-select

[![CI](https://github.com/keivanmalhani/shutter-select/actions/workflows/ci.yml/badge.svg)](https://github.com/keivanmalhani/shutter-select/actions/workflows/ci.yml)
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

[English](README.md) | Espanol

![Demo de shutter-select: un comando convierte una carpeta de material en una linea de tiempo de selects lista para editar](docs/demo.gif)

Un motor local de culling de video. Apuntalo a una carpeta de material crudo: transcribe cada palabra hablada, encuentra las tomas, marca el audio saturado o demasiado bajo, evalua nitidez, exposicion y movimiento, elige las mejores tomas y el mejor b-roll, y le entrega a tu editor una linea de tiempo de selects con marcadores de color, mas subtitulos SRT de todo lo dicho. Nada sale de tu maquina.

## Por que existe

Los editores pierden horas haciendo selects: recorrer entrevistas buscando la buena respuesta, cazar el b-roll nitido, descubrir tarde que la mejor toma tiene el audio saturado. Existen herramientas en la nube para partes de esto, pero el material crudo de un cliente no pertenece a servidores ajenos, y los detectores de escenas se quedan en listas de cortes sin terminar el trabajo. shutter-select cierra la brecha entre una tarjeta llena de material y el primer corte: todo corre en tu CPU y el resultado se abre directo en DaVinci Resolve o Premiere Pro.

## Que hace

```text
recorrer  ->  transcribir  ->  segmentar  ->  medir  ->  puntuar  ->  emitir
```

1. **Recorrer.** Descubre los videos bajo la carpeta raiz (seguro ante symlinks). Archivos ocultos, carpetas con `_` y carpetas llamadas "do not include" o "do not use" siempre se omiten.
2. **Transcribir.** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) corre localmente con deteccion de voz. Ingles y espanol son de primera clase; el idioma se detecta por archivo.
3. **Segmentar.** El habla es la estructura principal: una entrevista en tripie no tiene cortes visuales, asi que la deteccion de escenas sola no encuentra nada justo donde mas importa. El habla continua se vuelve una toma (pausas de 1.5 s o mas separan tomas) y solo las regiones sin habla se segmentan con los cortes de [PySceneDetect](https://github.com/Breakthrough/PySceneDetect). Cada segmento se clasifica como toma hablada o b-roll.
4. **Medir.** Por segmento: nivel de audio, saturacion, proporcion de silencio y un margen voz-ruido calculado de la onda; nitidez (varianza del Laplaciano), recorte de exposicion y movimiento sobre cuadros muestreados. La presencia de rostros es opcional (`--faces`).
5. **Puntuar.** Cada medida se normaliza por percentil dentro de la corrida y dentro de su clase, asi cada sesion se juzga contra si misma y nunca contra umbrales absolutos fragiles. Las tomas pesan mas el audio; el b-roll pesa mas la nitidez y el movimiento. El habla saturada o inaudible falla directo con la razon adjunta.
6. **Emitir.** Se marcan los picks claros y los descartes claros; el resto queda a tu criterio. Sale una linea de tiempo de selects, un stringout completo con marcadores verde/rojo/amarillo en cada segmento, subtitulos SRT por archivo, un reporte en texto plano y un JSON de medidas crudas por archivo para otras herramientas.

## Que obtienes

| Archivo | Que es |
| --- | --- |
| `selects.otio` | Las tomas y el b-roll elegidos como una linea de tiempo. Se abre nativo en DaVinci Resolve. |
| `stringout.otio` | Cada segmento en orden con su marcador verde, rojo o amarillo: todo el razonamiento del motor, visible en una linea de tiempo. |
| `selects.fcp7.xml` | La linea de tiempo de selects para importar en Premiere Pro. |
| `selects.edl` | Respaldo CMX 3600 de solo cortes (EDL casi no soporta marcadores; documentado, no disimulado). |
| `transcripts/*.srt` | Subtitulos de todo lo dicho, por archivo fuente. |
| `transcripts/transcript.json` | La transcripcion completa con tiempos (por palabra con `--words`). |
| `report.txt` | Cada decision con su razon, mas cuanto material realmente te queda por revisar. |
| `cache/*.json` | Medidas crudas por segmento y por archivo. Las corridas siguientes omiten archivos sin cambios; otras herramientas pueden reordenar sin re-analizar. |

## Instalacion

Requiere Python 3.11+ y [ffmpeg](https://ffmpeg.org/download.html).

```bash
brew install ffmpeg            # macOS
```

```bash
sudo apt install ffmpeg        # Debian/Ubuntu
```

Aun no esta en PyPI (ver Hoja de ruta). Instalacion desde el codigo:

```bash
git clone https://github.com/keivanmalhani/shutter-select.git
cd shutter-select
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Uso

Listar el material y sus formatos. Solo lectura, instantaneo:

```bash
shutter-select scan ~/material/sesion_canon
```

Primera corrida: analizar y emitir en un paso, permitiendo la descarga unica del modelo whisper:

```bash
shutter-select run ~/material/sesion_canon --allow-download
```

Reajustar umbrales al instante sin re-analizar (el analisis queda en cache por archivo):

```bash
shutter-select emit ~/material/sesion_canon --select-percentile 0.75
```

Despues abre `_selects/selects.otio` en Resolve (File > Import > Timeline) o `_selects/selects.fcp7.xml` en Premiere. Los marcadores llevan la puntuacion, las razones y un fragmento de la transcripcion.

### Opciones

| Bandera | Por defecto | Que hace |
| --- | --- | --- |
| `--model NAME` | small | Tamano de whisper: tiny, base, small, medium. |
| `--allow-download` | apagado | Permite la descarga unica del modelo, el unico acceso a red que existe. |
| `--profile P` | auto | Perfil de pesos: auto, interview, event, broll. |
| `--words` | apagado | Guarda tiempos por palabra (para herramientas de subtitulado). |
| `--faces` | apagado | Presencia de rostros via YuNet (descarga unica verificada con sha256). |
| `--take-gap S` | 1.5 | Pausa en segundos que separa dos tomas. |
| `--scene-threshold T` | 27 | Sensibilidad de PySceneDetect en regiones sin habla. |
| `--select-percentile` | 0.60 | Que tan exigente es el pick (mas alto = menos picks). |
| `--out DIR` | raiz/_selects | Donde se escribe todo. |

## Politica de pesos de modelos

Nada viene incluido en el repo y nada se descarga en silencio. Los modelos whisper se resuelven desde Hugging Face a un cache local la primera vez que pasas `--allow-download`; el detector de rostros es un solo archivo ONNX verificado contra el sha256 fijado en [MODEL_MANIFEST.json](MODEL_MANIFEST.json) antes de guardarse. Sin la bandera y sin modelo en cache, la herramienta te dice exactamente que hacer; nunca descarga por su cuenta.

## Modelo de seguridad

- **Solo local.** Sin subidas, sin cuentas, sin telemetria, sin analitica.
- **Sin red en tiempo de ejecucion.** La unica excepcion es la descarga de modelos explicita y opcional de arriba.
- **Las fuentes se abren solo en lectura.** Todo se escribe bajo `_selects/` (o `--out`); los archivos fuente nunca se modifican.
- **Recorridos seguros ante symlinks.** Un symlink plantado no puede jalar material de fuera de la raiz que nombraste.
- **Reglas de omision siempre activas.** Las carpetas llamadas "do not include" o "do not use" se excluyen a cualquier profundidad, asi una carpeta intocable en el disco de un cliente queda intocada.

## Limitaciones honestas

- El margen voz-ruido es una heuristica (nivel de las partes fuertes sobre el piso silencioso), no una medicion SNR calibrada.
- El volumen por segmento es RMS dBFS, no LUFS (el LUFS integrado por archivo si se guarda).
- Un corte de escena nunca divide una toma hablada continua en esta version.
- La puntuacion es una senal de ranking para acelerar tu criterio, no para reemplazarlo. El stringout existe para auditar cada decision.

## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

56 pruebas unitarias, sin material binario en el repo: el video de prueba se sintetiza con ffmpeg al momento. Una prueba de extremo a extremo corre transcripcion real detras de `-m integration` y queda fuera de CI. CI corre Python 3.11 y 3.12 con ffmpeg instalado.

## Familia

Parte de una familia de herramientas locales de foto y video: [shutter-cull](https://github.com/keivanmalhani/shutter-cull) hace culling de sesiones de foto hacia picks listos para Lightroom, [shutter-mcp](https://github.com/keivanmalhani/shutter-mcp) expone bibliotecas de fotos a agentes de IA en solo lectura. El JSON de analisis por archivo de shutter-select es un contrato estable (`schema_version`), disenado para alimentar el ranking de clips sociales que sigue.

## Hoja de ruta

- Muestreo de cuadros en una sola pasada para acelerar 4K
- Diarizacion de hablantes y deteccion de muletillas
- Exportes listos para subtitulos sobre `--words`
- Publicacion en PyPI
- Ranking de clips sociales sobre el JSON de analisis

## Licencia

MIT, ver [LICENSE](LICENSE).
