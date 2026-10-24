import argparse

from _header import print_header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    print_header(nonce=args.nonce, seed=42, script_path=__file__)


if __name__ == "__main__":
    main()
