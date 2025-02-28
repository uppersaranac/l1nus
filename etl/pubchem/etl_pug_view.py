# iterate over a user settable range of pubchem cids. For each cid, download via pubchem pug_view the cid summary in json format. Do not download faster than 5 cids per second. If pug_view returns an http error code or the http X-Throttling-Control header contains the strings yellow, red, or black, wait for 5 minutes before trying again. Save the json formatted summaries into files that contain sets of up to 10000 cids. Do not assume the cids are contiguous. The files should be json formatted themselves, with each cid record an element of an array. The files should be named using the start and end cid. Keep track in a state file the last successfully downloaded cid so that if the script is stopped and restarted, it will start with the next cid that needs to be downloaded. The script should be able to be run from the command line with the following arguments: start_cid, end_cid, output_dir, state_file, cids_per_file, cids_per_second, throttle_wait_time.

import os
import time
import requests
import json
import argparse
import gzip

def download_pubchem_summaries(start_cid, end_cid, output_dir, state_file, cids_per_file, cids_per_second, throttle_wait_time):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
    delay_between_requests = 1.0 / cids_per_second

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    last_cid = start_cid
    state_file = os.path.join(output_dir, state_file)
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            last_cid = int(f.read().strip()) + 1

    summaries = []
    for cid in range(last_cid, end_cid + 1):
        try:
            start_time = time.time()
            response = requests.get(base_url.format(cid))
            print(f"Downloading CID {cid} - {response.status_code}")
            elapsed_time = time.time() - start_time

            if response.status_code != 200 or any(throttle in response.headers.get('X-Throttling-Control', '').lower() for throttle in ['yellow', 'red', 'black']):
                print(f"Throttling detected or error {response.status_code}. Waiting for {throttle_wait_time} seconds.")
                time.sleep(throttle_wait_time)
                continue

            # if the pug_view response contains "Invalid record number", wait delay_between_requests and skip this CID
            if "Invalid record number" in response.text:
                print(f"Invalid record number for CID {cid}")
                time.sleep(max(0, delay_between_requests - elapsed_time))
                continue

            summaries.append(response.json())
            if len(summaries) >= cids_per_file:
                # do not assume the cids are continuous. You must keep track of the starting cid to name the file
                save_summaries(summaries, output_dir, summaries[0]['Record']['RecordNumber'], cid)
                with open(state_file, 'w') as f:
                    f.write(str(cid))
                summaries = []

            time.sleep(max(0, delay_between_requests - elapsed_time))

        except Exception as e:
            print(f"Error downloading CID {cid}: {e}")
            time.sleep(throttle_wait_time)

    if summaries:
        save_summaries(summaries, output_dir, summaries[0]['Record']['RecordNumber'], cid)
        with open(state_file, 'w') as f:
            f.write(str(cid))


def save_summaries(summaries, output_dir, start_cid, end_cid):
    file_path = os.path.join(output_dir, f"summaries_{start_cid}_{end_cid}.json.gz")
    with gzip.open(file_path, 'wt', encoding='UTF-8') as f:
        json.dump(summaries, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PubChem summaries.")
    parser.add_argument("--start_cid", type=int, default=1, help="Starting CID")
    parser.add_argument("--end_cid", type=int, default=10000, help="Ending CID")
    parser.add_argument("--output_dir", default='pubchem', type=str, help="Output directory")
    parser.add_argument("--state_file", type=str, default='pubchem_state', help="State file path")
    parser.add_argument("--cids_per_file", type=int, default=1000, help="Number of CIDs per file")
    parser.add_argument("--cids_per_second", type=int, default=1, help="Number of CIDs to download per second")
    parser.add_argument("--throttle_wait_time", type=int, default=300, help="Throttle wait time in seconds")

    args = parser.parse_args()

    download_pubchem_summaries(
        args.start_cid,
        args.end_cid,
        args.output_dir,
        args.state_file,
        args.cids_per_file,
        args.cids_per_second,
        args.throttle_wait_time
    )