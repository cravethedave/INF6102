set -e
source venv/bin/activate
export PYTHONPATH="/Users/benjamindjian/Documents/DD PolyMTL/Trimestre Hiver 2024 (H24)/INF6102/Devoirs/Devoir1/"

mkdir -p solutions/{easy,hard}

for instance_number in {1..5}; do
  if [[ ${#instance_number} -lt 2 ]]; then
    instance_number="0${instance_number}"
  fi
  python3 main.py --agent=advanced --group=easy --number="$instance_number" --visualisation_file="solutions/easy/in_$instance_number"
done

for instance_number in {1..10}; do
  if [[ ${#instance_number} -lt 2 ]]; then
    instance_number="0${instance_number}"
  fi
  python3 main.py --agent=advanced --group=hard --number="$instance_number" --visualisation_file="solutions/hard/in_$instance_number"
done
