// #define __USE_GNU

#include <stdio.h>

#include <signal.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <err.h>
#include <execinfo.h>
#include <fenv.h>
#include <unistd.h>
// #ifndef __USE_GNU
#define __USE_GNU 1
// #include <ucontext.h>
/* Resolve symbol name and source location given the path to the executable 
   and an address */
int addr2line(char const * const program_name, void const * const addr)
{
    char addr2line_cmd[512] = {0};
    printf("program_name %s, addr %p", program_name, addr);
    /* have addr2line map the address to the relent line in the code */
    if(program_name == NULL)
    {
        sprintf(addr2line_cmd,"addr2line -f -p %p", addr);
    }
    else
    {
        // printf("program_name %s, addr %p", program_name, addr);
        sprintf(addr2line_cmd,"addr2line -f -p -e %.256s %p", program_name, addr);
        // printf("addr2line_cmd %s", addr2line_cmd);
    }
    printf("addr2line_cmd %s", addr2line_cmd);
    return system(addr2line_cmd);
}

#define MAX_STACK_FRAMES 64
static void *stack_traces[MAX_STACK_FRAMES];
void posix_print_stack_trace()
{
    int i, trace_size = 0;
    char **messages = (char **)NULL;

    trace_size = backtrace(stack_traces, MAX_STACK_FRAMES);
    messages = backtrace_symbols(stack_traces, trace_size);

    /* skip the first couple stack frames (as they are this function and
        our handler) and also skip the last frame as it's (always?) junk. */
    for (i = 3; i < (trace_size - 1); ++i)
    {
        // if (addr2line(icky_global_program_name, stack_traces[i]) != 0)
        {
            printf("Stack Frame %d at %s\n", i, messages[i]);
        }

    }
    if (messages) { free(messages); } 
}

void posix_signal_handler(int sig, siginfo_t *siginfo, void *vcontext)
{

    // char output[100];
    // ucontext_t *context = (ucontext_t*)vcontext;
    // #ifdef __x86_64__
    // unsigned long pc = context->uc_mcontext.gregs[REG_RIP];
    // #else
    // unsigned long pc = context->uc_mcontext.gregs[REG_EIP];
    // #endif 
    // printf("Caught at %lx value %lx\n",pc,*(int*)pc);
    // printf("ptr->uc_mcontext.pc = %llx\n", context->uc_mcontext.pc);
    // context->uc_mcontext.pc += 4;
    // write(1,output,strlen(output)+1);
    // context->uc_mcontext.gregs[REG_PC] = context->uc_mcontext.gregs[REG_nPC];
    // context->uc_mcontext.gregs[REG_nPC] = context->uc_mcontext.gregs[REG_nPC]+4;
    // (void)context;
    switch(sig)
    {
    case SIGSEGV:
        fputs("Caught SIGSEGV: Segmentation Fault\n", stderr);
        break;
    case SIGINT:
        fputs("Caught SIGINT: Interactive attention signal, (usually ctrl+c)\n", stderr);
        break;
    case SIGFPE:
        switch(siginfo->si_code)
        {
        case FPE_INTDIV:
            fputs("Caught SIGFPE: (integer divide by zero)\n", stderr);
            break;
        case FPE_INTOVF:
            fputs("Caught SIGFPE: (integer overflow)\n", stderr);
            break;
        case FPE_FLTDIV:
            fputs("Caught SIGFPE: (floating-point divide by zero)\n", stderr);
            break;
        case FPE_FLTOVF:
            fputs("Caught SIGFPE: (floating-point overflow)\n", stderr);
            break;
        case FPE_FLTUND:
            fputs("Caught SIGFPE: (floating-point underflow)\n", stderr);
            break;
        case FPE_FLTRES:
            fputs("Caught SIGFPE: (floating-point inexact result)\n", stderr);
            break;
        case FPE_FLTINV:
            fputs("Caught SIGFPE: (floating-point invalid operation)\n", stderr);
            break;
        case FPE_FLTSUB:
            fputs("Caught SIGFPE: (subscript out of range)\n", stderr);
            break;
        default:
            fputs("Caught SIGFPE: Arithmetic Exception\n", stderr);
            break;
        }
    case SIGILL:
        switch(siginfo->si_code)
        {
        case ILL_ILLOPC:
            fputs("Caught SIGILL: (illegal opcode)\n", stderr);
            break;
        case ILL_ILLOPN:
            fputs("Caught SIGILL: (illegal operand)\n", stderr);
            break;
        case ILL_ILLADR:
            fputs("Caught SIGILL: (illegal addressing mode)\n", stderr);
            break;
        case ILL_ILLTRP:
            fputs("Caught SIGILL: (illegal trap)\n", stderr);
            break;
        case ILL_PRVOPC:
            fputs("Caught SIGILL: (privileged opcode)\n", stderr);
            break;
        case ILL_PRVREG:
            fputs("Caught SIGILL: (privileged register)\n", stderr);
            break;
        case ILL_COPROC:
            fputs("Caught SIGILL: (coprocessor error)\n", stderr);
            break;
        case ILL_BADSTK:
            fputs("Caught SIGILL: (internal stack error)\n", stderr);
            break;
        default:
            fputs("Caught SIGILL: Illegal Instruction\n", stderr);
            break;
        }
        break;
    case SIGTERM:
        fputs("Caught SIGTERM: a termination request was sent to the program\n", stderr);
        break;
    case SIGABRT:
        fputs("Caught SIGABRT: usually caused by an abort() or assert()\n", stderr);
        break;
    default:
        break;
    }
    posix_print_stack_trace();
    //   abort();
    _Exit(1);
}

static uint8_t alternate_stack[SIGSTKSZ];
void set_signal_handler()
{
  /* setup alternate stack */
  {
    stack_t ss = {};
    ss.ss_sp = (void*)alternate_stack;
    ss.ss_size = SIGSTKSZ;
    ss.ss_flags = 0;

    if (sigaltstack(&ss, NULL) != 0) { err(1, "sigaltstack"); }
  }

  /* register our signal handlers */
  {
    struct sigaction sig_action = {};
    sig_action.sa_sigaction = posix_signal_handler;
    sigemptyset(&sig_action.sa_mask);


    sig_action.sa_flags = SA_SIGINFO | SA_ONSTACK;

    // feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
    if (sigaction(SIGSEGV, &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGFPE,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGINT,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGILL,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGTERM, &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGABRT, &sig_action, NULL) != 0) { err(1, "sigaction"); }
  }
}
char const * icky_global_program_name;
int regSig()
{
  /* store off program path so we can use it later */
  icky_global_program_name = NULL;
  set_signal_handler();
  return 0;
}

int divide_by_zero()
{
  int a = 1;
  int b = 0; 
  return a / b;
}
void cause_segfault()
{
  int * p = (int*)0x12345678;
  *p = 0;
}
void stack_overflow()
{
  int foo[1000];
  (void)foo;
  stack_overflow();
}
/* break out with ctrl+c to test SIGINT handling */
void infinite_loop()
{
  while(1) {};
}
void illegal_instruction()
{
  /* I couldn't find an easy way to cause this one, so I'm cheating */
  raise(SIGILL);
}
void cause_calamity()
{
    divide_by_zero();
}
// #endif //__use_gnu