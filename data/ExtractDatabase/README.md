# Directory Structure
- include/		&emsp;Include directory.
- CreateDatabase.cpp &emsp;Extract subsets of specified sizes from a designated dataset—for example, extract subsets of 10,000; 20,000; ...; 200,000 from the IMDB dataset to evaluate the performance of TDPTC* across different data scales.
- Makefile		&emsp;Makefile.
- MemoryOperation.h		&emsp;Header for memory operation.
- mt19937ar.h		&emsp;Header for Mersenne Twister.

# Required Libraries
* StatsLib
  * https://www.kthohr.com/statslib.htm
* Gcem
  * https://github.com/kthohr/gcem/tree/master/include

# Setup
Please put the following files/directories under 'include/'.

**StatsLib**
- stats.hpp
- stats_incl/

**Gcem**
- gcem.hpp
- gcem_incl/
