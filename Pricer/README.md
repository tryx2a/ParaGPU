Pricer2014
==========

= Build Instructions =
  - Requires:
    - Cmake
    - g++
    - Boost
    - PNL
    - Google Tests

  - Steps:
    - cd /root/of/project
    - mkdir build
    - cd build
    - cmake -DPNL_ROOT=/path/to/PNL/build ..
    - make


= Execution Instructions =
  To use the pricer you can use the binary pricer:
    - pricer option.dat : print out the price of the option
    - pricer -c option.dat : print out the P&L of the option