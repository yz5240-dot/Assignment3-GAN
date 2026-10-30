"""Verify a running API over HTTP; the server can be started with Docker."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    base = args.url.rstrip('/')
    with urlopen(base + '/health', timeout=30) as response:
        health = json.load(response)
    assert health['model_file_present'], 'Trained weights are missing.'
    endpoint = base + '/generate?num_samples=16&seed=42'
    with urlopen(endpoint, timeout=60) as response:
        content_type = response.headers.get_content_type()
        png = response.read()
    assert content_type == 'image/png'
    assert png.startswith(b'\x89PNG\r\n\x1a\n')
    with urlopen(endpoint, timeout=60) as response:
        assert png == response.read(), 'Seeded images differ.'
    try:
        urlopen(base + '/generate?num_samples=0', timeout=30)
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError('An invalid sample count must return HTTP 422.')
    output = Path('outputs')
    output.mkdir(exist_ok=True)
    (output / 'http_api_generated.png').write_bytes(png)
    (output / 'http_api_test_results.json').write_text(json.dumps({
        'base_url': base, 'health': health, 'generation_status': 200,
        'content_type': content_type, 'same_seed_repeatable': True,
        'invalid_count_status': 422, 'test_method': 'Live HTTP requests',
        'note': 'These checks verify the HTTP service; use docker compose ps/logs to verify container deployment.'
    }, indent=2))
    print('PASS: health, generated PNG, repeatable seed, and invalid parameter handling.')
    print('Saved outputs/http_api_generated.png and outputs/http_api_test_results.json')


if __name__ == '__main__':
    main()
