//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch_test_macros.hpp"
#include "vector_ops.hpp"

TEST_CASE( "Mean of a vector is computed", "[mean]" ) {
    REQUIRE( re_error(0, mean({0,0,0,0,0,0,0,0,0})) <= 1e-3 );
}