import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Preprocessing data from {args.input} to {args.output}")

if __name__ == "__main__":
    main()
