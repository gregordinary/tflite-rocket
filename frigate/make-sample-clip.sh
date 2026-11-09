#!/usr/bin/env bash
# Generate media/sample.mp4 for the bring-up config: a still scene first, then a
# person walks through.
#
# Why the still-then-move shape: Frigate's motion detector (improved_motion) starts in
# a "calibrating" state and only leaves it once motion falls below ~5% of the frame
# with < 4 contours; while calibrating, the camera loop ignores all motion boxes, so no
# detection runs. A clip with *continuous* motion (a pan, or an object always moving)
# never lets calibration finish, so the detector never fires. The few still seconds at
# the start let calibration complete; the person crossing then triggers detection. A
# real camera gets still periods naturally, so it does not need this.
#
# Needs ffmpeg + curl. Writes ./media/sample.mp4.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p media && cd media

# Two Wikimedia Commons images (override BG_URL / FG_URL to use your own):
#   background  "Market-street-kearny-street-bus.jpg", User:Firstcultural, CC0 1.0
#   foreground  "Couple walking dog at Buckingham Fountain.jpg", public domain
# https://commons.wikimedia.org/wiki/File:<name> has each file's page.
COMMONS=https://commons.wikimedia.org/wiki/Special:FilePath
BG_URL="${BG_URL:-$COMMONS/Market-street-kearny-street-bus.jpg?width=1280}"
FG_URL="${FG_URL:-$COMMONS/Couple_walking_dog_at_Buckingham_Fountain.jpg?width=1280}"
[ -f bg.jpg ] || curl -fsSL -o bg.jpg "$BG_URL"
[ -f fg.jpg ] || curl -fsSL -o fg.jpg "$FG_URL"

# 15s: ~5s still (calibrate) -> a person sprite crosses left-to-right (detect) -> still.
ffmpeg -hide_banner -loglevel error -y -loop 1 -t 15 -i bg.jpg -loop 1 -t 15 -i fg.jpg \
  -filter_complex "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1,setsar=1[bg];\
[1:v]scale=440:-1,setsar=1[fg];\
[bg][fg]overlay=x=if(lt(t\,5)\,-600\,(t-5)*300-600):y=280:shortest=1,fps=15,format=yuv420p[v]" \
  -map "[v]" -c:v libx264 -preset veryfast -g 30 sample.mp4
echo "wrote $(pwd)/sample.mp4 ($(ffprobe -v error -show_entries format=duration -of csv=p=0 sample.mp4)s)"
