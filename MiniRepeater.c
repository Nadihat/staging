#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_INTENT_SIZE (32 * 1024)  // 32 KB static limit for intention text
#define MAX_INPUT_SIZE  (4 * 1024)   // 4 KB static limit for terminal input
#define MAX_PATH_SIZE   1024         // 1 KB static limit for file path

/* --- Static Memory Allocations (Total: ~37.1 KB) --- */
static char g_base_intent[MAX_INTENT_SIZE];
static char g_input_buffer[MAX_INPUT_SIZE];
static char g_file_path[MAX_PATH_SIZE];

static int g_repeats = 0;
static int g_duration = 0;
static int g_is_file = 0;

// Thread parameters structure
typedef struct {
    const char* base_intent;
    int repeats;
} thread_args_t;

static thread_args_t g_thread_args;
static volatile uint32_t g_last_hash = 0; // Volatile to prevent compiler optimization pruning

/* --- Utility Functions --- */

/* Safe conversion of uint32 to ASCII string representation without using dynamic allocation */
static inline int uint32_to_str(uint32_t val, char* buf) {
    if (val == 0) {
        buf[0] = '0';
        buf[1] = '\0';
        return 1;
    }
    char temp[12];
    int i = 0;
    while (val > 0) {
        temp[i++] = (char)('0' + (val % 10));
        val /= 10;
    }
    int len = i;
    for (int j = 0; j < len; j++) {
        buf[j] = temp[len - 1 - j];
    }
    buf[len] = '\0';
    return len;
}

/* Fast polynomial rolling hash algorithm matching python implementation */
static uint32_t compute_hash(const char* str, int len) {
    uint32_t h = 0;
    for (int i = 0; i < len; i++) {
        h = h * 31 + (uint8_t)str[i];
    }
    return h;
}

/* Portable file existence check */
int file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return 1;
    }
    return 0;
}

/* Strip leading and trailing whitespace characters */
void trim_whitespace(char* str) {
    size_t len = strlen(str);
    while (len > 0 && (str[len - 1] == ' ' || str[len - 1] == '\n' || 
                       str[len - 1] == '\r' || str[len - 1] == '\t')) {
        str[--len] = '\0';
    }
    char* start = str;
    while (*start == ' ' || *start == '\n' || *start == '\r' || *start == '\t') {
        start++;
    }
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

/* Read content from file safely into our static buffer */
void read_intention_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        exit(1);
    }
    size_t bytes_read = fread(g_base_intent, 1, MAX_INTENT_SIZE - 1, file);
    g_base_intent[bytes_read] = '\0';
    fclose(file);
    trim_whitespace(g_base_intent);
}

/* Format time as HH:MM:SS */
void format_time(int seconds, char* buf) {
    int hrs = seconds / 3600;
    int mins = (seconds % 3600) / 60;
    int secs = seconds % 60;
    sprintf(buf, "%02d:%02d:%02d", hrs, mins, secs);
}

/* Signal handler for handling Ctrl+C gracefully */
void sigint_handler(int sig) {
    (void)sig;
    printf("\n\nScript interrupted by user. Exiting gracefully.\n");
    exit(0);
}

/* Background worker thread performing the heavy hash calculation */
void* execute_intentions(void* arg) {
    thread_args_t* args = (thread_args_t*)arg;
    const char* base_intent = args->base_intent;
    int repeats = args->repeats;

    char num_buf[12];

    for (int r = 0; r < repeats; r++) {
        // First iteration of the loop hashes the base intention string
        uint32_t h = compute_hash(base_intent, strlen(base_intent));

        // Remaining 88,888,887 iterations hash the string conversion of the previous hash
        for (uint32_t iter = 1; iter < 88888888; iter++) {
            int len = uint32_to_str(h, num_buf);
            h = compute_hash(num_buf, len);
        }
        g_last_hash = h; // Write to volatile storage to prevent compiler dead-code elimination
    }
    return NULL;
}

/* Interactive helper functions */
int prompt_positive_int(const char* prompt_text) {
    char input_buf[128];
    while (1) {
        printf("%s", prompt_text);
        if (fgets(input_buf, sizeof(input_buf), stdin) == NULL) {
            continue;
        }
        trim_whitespace(input_buf);
        char* endptr;
        long val = strtol(input_buf, &endptr, 10);
        if (endptr == input_buf || *endptr != '\0' || val <= 0) {
            printf("Invalid input. Please enter a positive integer.\n");
            continue;
        }
        return (int)val;
    }
}

void prompt_intention_or_file() {
    while (1) {
        printf("Intention (or filename): ");
        if (fgets(g_input_buffer, sizeof(g_input_buffer), stdin) == NULL) {
            continue;
        }
        trim_whitespace(g_input_buffer);

        if (strlen(g_input_buffer) == 0) {
            printf("Input cannot be empty. Please enter a valid intention or filename.\n");
            continue;
        }

        if (file_exists(g_input_buffer)) {
            g_is_file = 1;
            strncpy(g_file_path, g_input_buffer, sizeof(g_file_path) - 1);
            g_file_path[sizeof(g_file_path) - 1] = '\0';
        } else {
            g_is_file = 0;
            strncpy(g_base_intent, g_input_buffer, sizeof(g_base_intent) - 1);
            g_base_intent[sizeof(g_base_intent) - 1] = '\0';
        }
        break;
    }
}

int is_positive_int(const char* str) {
    char* endptr;
    long val = strtol(str, &endptr, 10);
    if (endptr == str || *endptr != '\0' || val <= 0) {
        return 0;
    }
    return (int)val;
}

void print_help() {
    printf("Usage: servitor_connect [options]\n");
    printf("Options:\n");
    printf("  --file <path>       The file containing the intention.\n");
    printf("  --intent <string>   The intention string.\n");
    printf("  --repeats <int>     Number of times to repeat the intention.\n");
    printf("  --duration <int>    Duration in seconds to repeat the intentions for.\n");
    printf("  --help, -?          Show this help message and exit.\n");
}

/* --- Program Entry Point --- */
int main(int argc, char* argv[]) {
    signal(SIGINT, sigint_handler);

    printf("ServitorConnect CLI v3 (C Edition)\n");
    printf("by AnthroHeart/Anthro Teacher/Thomas Sweet\n\n");

    // Parse CLI arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            strncpy(g_file_path, argv[++i], sizeof(g_file_path) - 1);
            g_file_path[sizeof(g_file_path) - 1] = '\0';
            g_is_file = 1;
        } else if (strcmp(argv[i], "--intent") == 0 && i + 1 < argc) {
            strncpy(g_base_intent, argv[++i], sizeof(g_base_intent) - 1);
            g_base_intent[sizeof(g_base_intent) - 1] = '\0';
            g_is_file = 0;
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            int val = is_positive_int(argv[++i]);
            if (val <= 0) {
                fprintf(stderr, "Invalid positive int value for --repeats: '%s'\n", argv[i]);
                exit(1);
            }
            g_repeats = val;
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            int val = is_positive_int(argv[++i]);
            if (val <= 0) {
                fprintf(stderr, "Invalid positive int value for --duration: '%s'\n", argv[i]);
                exit(1);
            }
            g_duration = val;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-?") == 0) {
            print_help();
            return 0;
        }
    }

    // Handle interactive input if parameters are missing
    if (strlen(g_file_path) == 0 && strlen(g_base_intent) == 0) {
        prompt_intention_or_file();
    }

    if (g_is_file) {
        if (!file_exists(g_file_path)) {
            fprintf(stderr, "File '%s' does not exist. Exiting.\n", g_file_path);
            exit(1);
        }
        read_intention_from_file(g_file_path);
    }

    if (g_repeats <= 0) {
        g_repeats = prompt_positive_int("Number of repeats per hour: ");
    }

    if (g_duration <= 0) {
        g_duration = prompt_positive_int("Duration in seconds: ");
    }

    // Output configuration parameters
    if (g_is_file) {
        printf("\nSource: File '%s'\n", g_file_path);
    } else {
        printf("\nSource: Intent '%s'\n", g_base_intent);
    }
    printf("Repeats per Hour: %d\n", g_repeats);
    
    char timer_str[16];
    format_time(g_duration, timer_str);
    printf("Duration: %s\n", timer_str);

    // Timing state variables
    time_t start_time = time(NULL);
    time_t end_time = start_time + g_duration;
    time_t next_hour_time = start_time + 3600;

    g_thread_args.base_intent = g_base_intent;
    g_thread_args.repeats = g_repeats;

    printf("\nRepeating Intention...\n");
    pthread_t thread_id;
    if (pthread_create(&thread_id, NULL, execute_intentions, &g_thread_args) != 0) {
        fprintf(stderr, "Error creating initial execution thread.\n");
        return 1;
    }
    pthread_detach(thread_id);

    // Main event loop
    while (1) {
        time_t current_time = time(NULL);

        if (current_time >= end_time) {
            break;
        }

        if (current_time >= next_hour_time) {
            printf("\nRepeating Intention...\n");
            if (pthread_create(&thread_id, NULL, execute_intentions, &g_thread_args) != 0) {
                fprintf(stderr, "Error creating background execution thread.\n");
            } else {
                pthread_detach(thread_id);
            }
            next_hour_time += 3600;
        }

        int remaining = (int)(end_time - current_time);
        if (remaining < 0) remaining = 0;

        format_time(remaining, timer_str);

        if (g_is_file) {
            printf("\rFile '%s' Repeated %d Times Hourly: %s", g_file_path, g_repeats, timer_str);
        } else {
            printf("\rIntent '%s' Repeated %d Times Hourly: %s", g_base_intent, g_repeats, timer_str);
        }
        fflush(stdout);

        sleep(1);
    }

    printf("\nDuration completed.\n");
    return 0;
}
