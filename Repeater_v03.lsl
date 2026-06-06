// ServitorConnect CLI v3 - LSL Edition (Fixed)
// Fixes: non-blocking chunked execution, threshold-based hourly trigger,
//        user input for all parameters, hover text cleanup, progress display.

// ─── Configuration ───────────────────────────────────────────────────────────
integer TOTAL_ITERATIONS = 88888;  // Platform-constrained (Python uses 88,888,888)
integer BATCH_SIZE = 500;          // Iterations per timer tick (tune for sim load)
float   TICK_INTERVAL = 0.1;      // Timer interval in seconds

// ─── User Parameters ─────────────────────────────────────────────────────────
string  g_base_intent = "";
integer g_repeats = 1;
integer g_duration = 3600;

// ─── Timing State ────────────────────────────────────────────────────────────
integer g_start_time;
integer g_end_time;
integer g_next_hour_time;  // Threshold-based hourly trigger (not modulo)

// ─── Chunked Computation State ───────────────────────────────────────────────
integer g_is_computing = FALSE;
integer g_current_iter = 0;
integer g_current_repeat = 0;
string  g_current_intention_value;

// ─── Dialog State ────────────────────────────────────────────────────────────
integer g_dialog_channel = -991827;
integer g_listener;
integer g_prompt_stage = 0;  // 0=intention, 1=repeats, 2=duration

// ─── Helper: Unsigned String Representation ──────────────────────────────────
string to_unsigned_string(integer val) {
    if (val >= 0) {
        return (string)val;
    }
    integer q = (val / 10) + 429496729;
    integer r = (val % 10) + 6;
    if (r >= 10) {
        q += 1;
        r -= 10;
    } else if (r < 0) {
        q -= 1;
        r += 10;
    }
    return (string)q + (string)r;
}

// ─── Helper: Polynomial Rolling Hash ─────────────────────────────────────────
integer compute_rolling_hash(string text) {
    integer h = 0;
    integer i;
    integer len = llStringLength(text);
    for (i = 0; i < len; ++i) {
        h = (h * 31) + llOrd(text, i);
    }
    return h;
}

// ─── Helper: Zero-Padded Two-Digit String ────────────────────────────────────
string pad_two(integer n) {
    if (n < 10) return "0" + (string)n;
    return (string)n;
}

// ─── Start Non-Blocking Computation ──────────────────────────────────────────
start_computation() {
    g_is_computing = TRUE;
    g_current_iter = 0;
    g_current_repeat = 0;
    g_current_intention_value = g_base_intent;
}

// ─── Dialog Prompt Dispatcher ────────────────────────────────────────────────
trigger_prompt() {
    llListenRemove(g_listener);
    g_listener = llListen(g_dialog_channel, "", llGetOwner(), "");

    if (g_prompt_stage == 0) {
        llTextBox(llGetOwner(),
            "ServitorConnect CLI v3\n\nEnter your intention string:",
            g_dialog_channel);
    } else if (g_prompt_stage == 1) {
        llTextBox(llGetOwner(),
            "Enter number of repeats per hour\n(positive integer):",
            g_dialog_channel);
    } else if (g_prompt_stage == 2) {
        llTextBox(llGetOwner(),
            "Enter duration in seconds\n(positive integer):",
            g_dialog_channel);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DEFAULT STATE — Collects user input via sequential dialog prompts
// ═══════════════════════════════════════════════════════════════════════════════
default {
    state_entry() {
        llSetText("", ZERO_VECTOR, 0.0);  // Clear stale hover text
        llOwnerSay("ServitorConnect LSL Script Initialized.");
        llOwnerSay("Touch this object to configure and start.");
        g_prompt_stage = 0;
        trigger_prompt();
    }

    touch_start(integer total_number) {
        if (llDetectedKey(0) == llGetOwner()) {
            g_prompt_stage = 0;
            trigger_prompt();
        }
    }

    listen(integer channel, string name, key id, string message) {
        if (channel != g_dialog_channel || id != llGetOwner()) return;

        string trimmed = llStringTrim(message, STRING_TRIM);

        if (g_prompt_stage == 0) {
            // ── Intention Input ──
            if (trimmed == "") {
                llOwnerSay("Intention cannot be empty. Try again.");
                trigger_prompt();
                return;
            }
            g_base_intent = trimmed;
            llOwnerSay("Intention accepted: '" + g_base_intent + "'");
            g_prompt_stage = 1;
            trigger_prompt();

        } else if (g_prompt_stage == 1) {
            // ── Repeats Input ──
            integer val = (integer)trimmed;
            if (val <= 0) {
                llOwnerSay("Please enter a positive integer for repeats.");
                trigger_prompt();
                return;
            }
            g_repeats = val;
            llOwnerSay("Repeats per hour: " + (string)g_repeats);
            g_prompt_stage = 2;
            trigger_prompt();

        } else if (g_prompt_stage == 2) {
            // ── Duration Input ──
            integer val = (integer)trimmed;
            if (val <= 0) {
                llOwnerSay("Please enter a positive integer for duration (seconds).");
                trigger_prompt();
                return;
            }
            g_duration = val;
            llOwnerSay("Duration: " + (string)g_duration + " seconds.");
            llListenRemove(g_listener);
            state running;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RUNNING STATE — Non-blocking chunked execution with responsive UI
// ═══════════════════════════════════════════════════════════════════════════════
state running {
    state_entry() {
        g_start_time = llGetUnixTime();
        g_end_time = g_start_time + g_duration;
        g_next_hour_time = g_start_time + 3600;  // First hourly mark

        llOwnerSay("Starting repetition cycle.");
        llOwnerSay("  Intent: " + g_base_intent);
        llOwnerSay("  Repeats/Hour: " + (string)g_repeats);
        llOwnerSay("  Duration: " + (string)g_duration + "s");

        // Begin first computation (non-blocking via chunked timer)
        start_computation();
        llSetTimerEvent(TICK_INTERVAL);
    }

    touch_start(integer total_number) {
        if (llDetectedKey(0) == llGetOwner()) {
            llOwnerSay("Resetting script. Re-opening input dialog.");
            llSetTimerEvent(0.0);
            g_is_computing = FALSE;
            llSetText("", ZERO_VECTOR, 0.0);
            state default;
        }
    }

    timer() {
        integer now = llGetUnixTime();

        // ── Check Duration Expiry ──
        if (now >= g_end_time) {
            llSetTimerEvent(0.0);
            g_is_computing = FALSE;
            llSetText("", ZERO_VECTOR, 0.0);
            llOwnerSay("Duration completed.");
            state default;
            return;
        }

        // ── Chunked Computation (non-blocking) ──
        if (g_is_computing) {
            integer end_iter = g_current_iter + BATCH_SIZE;
            if (end_iter > TOTAL_ITERATIONS) end_iter = TOTAL_ITERATIONS;

            integer i;
            for (i = g_current_iter; i < end_iter; ++i) {
                integer h = compute_rolling_hash(g_current_intention_value);
                g_current_intention_value = to_unsigned_string(h);
            }
            g_current_iter = end_iter;

            // Check if current repeat is complete
            if (g_current_iter >= TOTAL_ITERATIONS) {
                g_current_repeat += 1;
                llOwnerSay("Completed repeat " + (string)g_current_repeat +
                           "/" + (string)g_repeats +
                           ". Final Hash: " + g_current_intention_value);

                if (g_current_repeat >= g_repeats) {
                    // All repeats done for this cycle
                    g_is_computing = FALSE;
                } else {
                    // Start next repeat
                    g_current_iter = 0;
                    g_current_intention_value = g_base_intent;
                }
            }
        }

        // ── Hourly Retrigger (threshold-based, not modulo) ──
        if (!g_is_computing && now >= g_next_hour_time && now < g_end_time) {
            // Advance to next future hour (handles case where computation took > 1hr)
            while (g_next_hour_time <= now) {
                g_next_hour_time += 3600;
            }
            llOwnerSay("Hourly event triggered. Repeating intentions...");
            start_computation();
        }

        // ── UI Hover-Text Update ──
        integer remaining = g_end_time - now;
        if (remaining < 0) remaining = 0;
        integer hrs = remaining / 3600;
        integer mins = (remaining % 3600) / 60;
        integer secs = remaining % 60;

        string status;
        if (g_is_computing) {
            integer pct = (g_current_iter * 100) / TOTAL_ITERATIONS;
            status = "Computing [" + (string)pct + "%] "
                   + "Rep " + (string)(g_current_repeat + 1) + "/" + (string)g_repeats;
        } else {
            status = "Idle (next cycle at hourly mark)";
        }

        string display_text = "═══ ServitorConnect v3 ═══\n" +
                              "Status: " + status + "\n" +
                              "Intent: " + g_base_intent + "\n" +
                              "Repeats/Hour: " + (string)g_repeats + "\n" +
                              "Remaining: " + pad_two(hrs) + ":" +
                              pad_two(mins) + ":" + pad_two(secs) + "\n" +
                              "(Touch to Reset)";

        llSetText(display_text, <0.0, 1.0, 0.0>, 1.0);
    }
}
