#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fenv.h>
// #define FE_EXCEPT_SHIFT 22  // shift flags right to get masks
// #define FM_ALL_EXCEPT    FE_ALL_EXCEPT >> FE_EXCEPT_SHIFT 

#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

#ifdef _WIN32
  #include <windows.h>
  #include <imagehlp.h>
#else
  #include <err.h>
  #include <execinfo.h>
#endif


static char const * icky_global_program_name;

/* Resolve symbol name and source location given the path to the executable 
   and an address */
int addr2line(char const * const program_name, void const * const addr)
{
  char addr2line_cmd[512] = {0};
  printf("program_name %s, addr %p", program_name, addr);
  /* have addr2line map the address to the relent line in the code */
  #ifdef __APPLE__
    /* apple does things differently... */
    sprintf(addr2line_cmd,"atos -o %.256s %p", program_name, addr); 
  #else
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
    // sprintf(addr2line_cmd,"addr2line -f -p -e %.256s %p", program_name, addr); 
  #endif

  return system(addr2line_cmd);
}

static char *fe_code_name[] = {
  "FPE_NOOP",
  "FEP_DIVIDEBYZERO", "FPE_INVALID", "FPE_OVERFLOW", "FPE_FLTUND",
  "FPE_FLTRES", "FPE_FLTSUB", "FPE_INTDIV", "FPE_INTOVF"
  "FPE_UNKNOWN"
};

static void
fhdl ( int sig, siginfo_t *sip, ucontext_t *scp )
{
  int fe_code = sip->si_code;
  unsigned int excepts = fetestexcept (FE_ALL_EXCEPT);

  switch (fe_code)
  {
    case FPE_FLTDIV: fe_code = 1; break; // divideByZero
    case FPE_FLTINV: fe_code = 2; break; // invalid
    case FPE_FLTOVF: fe_code = 3; break; // overflow
    case FPE_FLTUND: fe_code = 4; break; // underflow
    case FPE_FLTRES: fe_code = 5; break; // inexact
    case FPE_FLTSUB: fe_code = 6; break; // invalid
    case FPE_INTDIV: fe_code = 7; break; // overflow
    case FPE_INTOVF: fe_code = 8; break; // underflow
            default: fe_code = 9;
   }

  if ( sig == SIGFPE )
  {
    printf ("signal:  SIGFPE with code %s\n", fe_code_name[fe_code]);
    // printf ("invalid flag:    0x%04X\n", excepts & FE_INVALID);
    // printf ("divByZero flag:  0x%04X\n", excepts & FE_DIVBYZERO);
    abort();
  }
  else printf ("Signal is not SIGFPE, it's %i.\n", sig);
  //feclearexcept(excepts);
  // abort();
  exit(-1);
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
  // for (i = 0; i &lt; trace_size; ++i) // we'll use this for now so you can see what's going on
  {
    if (addr2line(icky_global_program_name, stack_traces[i]) != 0)
    {
      printf("  error determining line # for: %s\n", messages[i]);
    }

  }
  if (messages) { free(messages); } 
}


void posix_signal_handler(int sig, siginfo_t *siginfo, void *vcontext)
{
  // (void)context;
  ucontext_t *context = (ucontext_t*)vcontext;
  #ifdef __x86_64__
  unsigned long pc = context->uc_mcontext.gregs[REG_RIP];
  #else
  unsigned long pc = context->uc_mcontext.gregs[REG_EIP];
  #endif 
  // void *eip[1] = { (void*)regs[REG_EIP] };

  char **symbol = backtrace_symbols(pc, 1);

  printf("Registers:eip is at %s\n", *symbol);

  // backtrace_symbols_fd(eip, 1, fd->_fileno);


  printf("Caught at %lx value %lx\n",pc,*(int*)pc);
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
  _Exit(1);
}

static uint8_t alternate_stack[SIGSTKSZ];
void set_signal_handler()
{
  /* setup alternate stack */
  {
    stack_t ss = {};
    /* malloc is usually used here, I'm not 100% sure my static allocation
       is valid but it seems to work just fine. */
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
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

    if (sigaction(SIGSEGV, &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGFPE,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGINT,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGILL,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGTERM, &sig_action, NULL) != 0) { err(1, "sigaction"); }
    if (sigaction(SIGABRT, &sig_action, NULL) != 0) { err(1, "sigaction"); }
  }
}
int regSig2(const char * program);

int main(int argc, char * argv[])
{
  (void)argc;

  /* store off program path so we can use it later */
  icky_global_program_name = argv[0];

  regSig2(argv[0]);
  double s = 1.0/0.0;
  puts("OMG! Nothing bad happend!");

  return 0;
}

int regSig2(const char * program)
{
  // (void)argc;

  /* store off program path so we can use it later */
  icky_global_program_name = program;
  printf("program_name %s\n", icky_global_program_name);
  set_signal_handler();

  // cause_calamity();

  // puts("OMG! Nothing bad happend!");

  return 0;
}

int regSig()
{
  double s;
  struct sigaction act;

  act.sa_sigaction = (void(*))fhdl;
  sigemptyset (&act.sa_mask);
  act.sa_flags = SA_SIGINFO;
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  // set handler
  if (sigaction(SIGFPE, &act, (struct sigaction *)0) != 0)
  {
      perror("shaohua");
      exit(-1);
  }
  return 0;
}