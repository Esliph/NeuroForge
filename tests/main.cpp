#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

int main(int argc, char** argv) {
  doctest::Context context;

  context.setOption("order-by", "name");
  context.setOption("no-breaks", true);
  context.applyCommandLine(argc, argv);

  int res = context.run();

  if (context.shouldExit())
    return res;

  int client_stuff_return_code = 0;

  return res + client_stuff_return_code;
}
