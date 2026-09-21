############################################################################
#   Helper functions to switch between types of coordinates                #
#                                                                          #
############################################################################

cdef location_t calculate_board_location(location_t x, location_t y, location_t size):
    """Return location on board
       no checks on outside board
       x = columns
       y = rows
    """

    # Return board location
    return x + (y * size)


cdef location_t calculate_board_location_or_border(location_t x, location_t y, location_t size):
    """Return location on board or borderlocation
       board locations = [0, size * size)
       border location = size * size
       x = columns
       y = rows
    """

    # Check if x or y are outside board
    if x < 0 or y < 0 or x >= size or y >= size:
        # Return border location
        return size * size

    # Return board location
    return calculate_board_location(x, y, size)


cdef tuple calculate_tuple_location(location_t location, location_t size):
    """1d index to 2d tuple location (x, y). Inverse of calculate_board_location()

       No sanity checks on bounds.
    """
    return divmod(location, size)

############################################################################
#   Neighbor/pattern lookup table creation functions                       #
#                                                                          #
############################################################################

cdef pattern_t get_neighbors(location_t size):
    """Create array for every board location with all 4 direct neighbor locations
       neighbor order: left - right - above - below

                -1     x
                      x x
                +1     x

                order:
                -1     2
                      0 1
                +1     3

       TODO neighbors is obsolete as neighbor3x3 contains the same values
    """

    # Initialize empty vector
    cdef pattern_t neighbor = pattern_t(4 * size * size)

    cdef short location
    cdef location_t x, y

    # Add all direct neighbors to every board location
    for y in range(size):
        for x in range(size):
            location = 4 * calculate_board_location(x, y, size)
            neighbor[location + 0] = calculate_board_location_or_border(x - 1, y, size)
            neighbor[location + 1] = calculate_board_location_or_border(x + 1, y, size)
            neighbor[location + 2] = calculate_board_location_or_border(x, y - 1, size)
            neighbor[location + 3] = calculate_board_location_or_border(x, y + 1, size)

    return neighbor


cdef pattern_t get_3x3_neighbors(location_t size):
    """Create for every board location array with all 8 surrounding neighbor locations
       neighbor order: above middle - middle left - middle right - below middle
                       above left - above right - below left - below right
                       this order is more useful as it separates neighbors and then diagonals
                -1    xxx
                      x x
                +1    xxx

                order:
                -1    405
                      1 2
                +1    637

        0-3 contains neighbors
        4-7 contains diagonals
    """

    # Initialize empty vector
    cdef pattern_t neighbor3x3 = pattern_t(8 * size * size)

    cdef short location
    cdef location_t x, y

    # Add all surrounding neighbors to every board location
    for x in range(size):
        for y in range(size):
            location = 8 * calculate_board_location(x, y, size)
            # Cardinal directions
            neighbor3x3[location + 0] = calculate_board_location_or_border(x, y - 1, size)
            neighbor3x3[location + 1] = calculate_board_location_or_border(x - 1, y, size)
            neighbor3x3[location + 2] = calculate_board_location_or_border(x + 1, y, size)
            neighbor3x3[location + 3] = calculate_board_location_or_border(x, y + 1, size)
            # Diagonal directions
            neighbor3x3[location + 4] = calculate_board_location_or_border(x - 1, y - 1, size)
            neighbor3x3[location + 5] = calculate_board_location_or_border(x + 1, y - 1, size)
            neighbor3x3[location + 6] = calculate_board_location_or_border(x - 1, y + 1, size)
            neighbor3x3[location + 7] = calculate_board_location_or_border(x + 1, y + 1, size)

    return neighbor3x3
