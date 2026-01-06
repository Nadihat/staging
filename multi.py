import time
import hashlib
import multiprocessing

# --- CONFIGURATION ---
NUM_INTENTIONS = 100
FREQ = 7.83
INTERVAL = 1 / FREQ
# Number of worker processes (usually match your CPU cores)
NUM_WORKERS = 4 

def sha3_512_hash(input_string):
    """The actual CPU work."""
    sha3 = hashlib.sha3_512()
    sha3.update(input_string.encode('utf-8'))
    return sha3.hexdigest()

def worker_task(intention_subset):
    """A worker receives a list of intentions and processes them."""
    return [sha3_512_hash(i) for i in intention_subset]

if __name__ == "__main__":
    # 1. Initialize our 100 intentions
    intentions = ["I am love " + str(i) for i in range(NUM_INTENTIONS)]
    
    # 2. Start the Worker Pool (The "Method 5" Workers)
    # These stay alive in the background, consuming very little RAM
    pool = multiprocessing.Pool(processes=NUM_WORKERS)

    print(f"Running {NUM_INTENTIONS} intentions via {NUM_WORKERS} workers at {FREQ}Hz")

    try:
        while True:
            start_time = time.perf_counter()

            # 3. Split the 100 intentions into batches for the workers
            # For 100 intentions and 4 workers, this makes 4 batches of 25
            chunk_size = len(intentions) // NUM_WORKERS
            chunks = [intentions[i:i + chunk_size] for i in range(0, len(intentions), chunk_size)]

            # 4. Dispatch work to the pool (Method 5 logic)
            # map() sends the tasks to the workers and waits for the batch to finish
            results = pool.map(worker_task, chunks)
            
            # Flatten the results back into our main list
            intentions = [item for sublist in results for item in sublist]

            # 5. Precise Timing
            elapsed = time.perf_counter() - start_time
            sleep_time = INTERVAL - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: CPU work took {elapsed:.4f}s, which is slower than the interval!")

    except KeyboardInterrupt:
        pool.terminate()
        print("Stopped.")
