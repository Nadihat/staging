; File: asm.asm
; Description: A countdown timer that, on an hourly basis, assigns the memory address of a user-provided
;              intention string to a register N times.
; NASM syntax for x86-64 Linux.
;
; To compile and run:
; nasm -f elf64 asm.asm -o asm.o
; gcc asm.o -o asm_runner -no-pie
; ./asm_runner "Your intention here" 1000

extern printf, sleep, fflush, atoi

section .data
    duration dq 3600      ; Total duration in seconds (1h)
    hourly_interval dq 3600 ; Repeat interval in seconds

    ; Format strings
    usage_msg db "Usage: %s <intention_string> <repeats_per_hour>", 10, 0
    hourly_trigger_msg db 10, "Hourly repeat triggered.", 10, 0
    timer_format db 13, "Time remaining: %02ld:%02ld:%02ld", 0
    done_msg db 10, "Duration completed.", 10, 0

section .bss
    remaining_seconds resq 1
    next_repeat_seconds resq 1
    intention_ptr resq 1
    repeats_count resq 1

section .text
    global main

main:
    ; Prologue
    push rbp
    mov rbp, rsp

    ; --- Argument Handling ---
    cmp rdi, 3          ; Check if argc is less than 3
    jl .usage_error     ; If so, print usage and exit

    ; Store intention string pointer from argv[1]
    mov rax, [rsi + 8]  ; rsi is argv, so [rsi+8] is argv[1]
    mov [intention_ptr], rax

    ; Parse and store repeats_per_hour from argv[2]
    mov rdi, [rsi + 16] ; rdi gets argv[2]
    call atoi           ; rax now holds the integer value
    mov [repeats_count], rax

    ; --- Initialize Timers ---
    mov rax, [duration]
    mov [remaining_seconds], rax
    mov rax, [hourly_interval]
    mov [next_repeat_seconds], rax

    ; Trigger the assignment loop at the beginning
    call repeat_intention_assignment

.loop_start:
    ; Check if total duration has run out
    mov r12, [remaining_seconds]
    cmp r12, 0
    jl .loop_end

    ; Display the countdown timer
    mov rdi, r12
    call format_and_print

    ; Check if it's time to repeat the intention
    mov r13, [next_repeat_seconds]
    cmp r13, 0
    jle .trigger_repeat ; Use jle to be safe

.after_repeat_check:
    ; --- Call sleep(1) ---
    mov rdi, 1
    xor rax, rax
    call sleep

    ; --- Decrement timers ---
    dec qword [remaining_seconds]
    dec qword [next_repeat_seconds]
    jmp .loop_start

.trigger_repeat:
    call repeat_intention_assignment
    ; Reset the hourly timer
    mov rax, [hourly_interval]
    mov [next_repeat_seconds], rax
    jmp .after_repeat_check

.usage_error:
    mov rdi, usage_msg
    mov rsi, [rsi]      ; rsi is argv, [rsi] is argv[0] (program name)
    xor rax, rax
    call printf
    mov rdi, 1          ; Exit code 1
    jmp .exit

.loop_end:
    ; Print the final "completed" message
    mov rdi, done_msg
    xor rax, rax
    call printf
    mov rdi, 0          ; Exit code 0

.exit:
    ; Epilogue and exit
    mov rsp, rbp
    pop rbp
    ret

;----------------------------------------------------
; repeat_intention_assignment:
; Mimics the assignment loop from S.py.
;----------------------------------------------------
repeat_intention_assignment:
    push rbp
    mov rbp, rsp

    ; Print a message indicating the hourly trigger fired
    mov rdi, hourly_trigger_msg
    xor rax, rax
    call printf

    ; The assignment loop
    mov rcx, [repeats_count] ; rcx is the loop counter
    cmp rcx, 0
    jle .assignment_loop_end

    mov rbx, [intention_ptr] ; rbx holds the pointer to the intention string

.assignment_loop:
    mov rax, rbx             ; The "assignment" operation: rax = intention_ptr
    loop .assignment_loop    ; Decrement rcx and jump if rcx is not 0

.assignment_loop_end:
    mov rsp, rbp
    pop rbp
    ret

;----------------------------------------------------
; format_and_print:
; Formats seconds into HH:MM:SS and prints to stdout.
; Input: rdi = total seconds
;----------------------------------------------------
format_and_print:
    push rbp
    mov rbp, rsp
    push rdi ; Save total seconds

    mov rax, rdi        ; rax = total_seconds
    xor rdx, rdx
    mov rbx, 3600
    div rbx             ; rax = hours, rdx = remainder

    push rax            ; Save hours
    mov rax, rdx        ; rax = remainder
    xor rdx, rdx
    mov rbx, 60
    div rbx             ; rax = minutes, rdx = seconds

    mov rcx, rdx        ; 4th arg for printf (seconds)
    mov rdx, rax        ; 3rd arg for printf (minutes)
    pop rsi             ; 2nd arg for printf (hours)
    mov rdi, timer_format ; 1st arg for printf (format string)

    xor rax, rax
    call printf

    ; Flush stdout
    xor rax, rax
    mov rdi, 0
    call fflush

    pop rdi ; Restore total seconds
    mov rsp, rbp
    pop rbp
    ret
