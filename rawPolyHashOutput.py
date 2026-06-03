import argparse
import sys
import time
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Computes the native 32-bit rolling polynomial hash of an intention over N iterations.',
        add_help=False
    )

    # Define mutually exclusive group for --file and --intent
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--file', type=str, help='The file containing the intention.')
    group.add_argument('--intent', type=str, help='The intention string.')

    def positive_int(value):
        try:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(f"Invalid positive int value: '{value}'")
            return ivalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")

    parser.add_argument('--iterations', '-n', type=positive_int, help='Number of times to hash the intention (N).')

    # Add help
    parser.add_argument('--help', '-?', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    return parser.parse_args()

def read_intention_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read().strip()
            return content
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

def prompt_intention_or_file():
    while True:
        user_input = input("Intention (or filename): ").strip()
        if not user_input:
            print("Input cannot be empty. Please enter a valid intention or filename.")
            continue
        if os.path.isfile(user_input):
            return {'file': user_input}
        else:
            return {'intent': user_input}

def prompt_positive_int(prompt_text):
    while True:
        user_input = input(prompt_text).strip()
        try:
            value = int(user_input)
            if value <= 0:
                print("Please enter a positive integer.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

def main():
    args = parse_arguments()
    print("Intention Polynomial Hasher v1")
    print("------------------------------\n")

    # Initialize variables
    file = args.file
    intent = args.intent
    iterations = args.iterations

    # Handle Intention or File
    if not (file or intent):
        user_input = prompt_intention_or_file()
        file = user_input.get('file')
        intent = user_input.get('intent')

    # Ensure that either file or intent is set
    if file:
        if not os.path.isfile(file):
            print(f"File '{file}' does not exist. Exiting.")
            sys.exit(1)
        source = f"File '{file}'"
        base_intent = read_intention_from_file(file)
    else:
        source = f"Intent '{intent}'"
        base_intent = intent

    # Handle Iterations
    if iterations is None:
        iterations = prompt_positive_int("Number of iterations (N): ")

    print(f"\nSource: {source}")
    print(f"Target Iterations: {iterations:,}")
    print("\nStarting calculation (Press Ctrl+C to abort)...\n")

    start_time = time.time()
    intention_value = base_intent

    try:
        # Loop exactly N times, chaining the hash output as the next input
        for i in range(1, iterations + 1):
            h = 0
            # Math: (hash * 31 + charCode) & 0xFFFFFFFF
            for char in str(intention_value):
                h = (h * 31 + ord(char)) & 0xFFFFFFFF
            
            intention_value = h

            # Update the progress tracker every 500,000 iterations to save CPU time on printing
            if i % 500000 == 0 or i == iterations:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\rProgress: {i:,} / {iterations:,} iterations complete... ({rate:,.0f} hashes/sec)")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nCalculation interrupted by user. Exiting.")
        sys.exit(0)

    elapsed_time = time.time() - start_time

    # Output the final results
    print("\n\n--- CALCULATION COMPLETE ---")
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"Final Polynomial Hash Output (Base-10): {intention_value}")
    
    # Also provide the corresponding 8-character Hex (like IntentionColor)
    hex_output = f"{intention_value:08X}"
    print(f"Final Polynomial Hash Output (Base-16): {hex_output}")

if __name__ == "__main__":
    main()
