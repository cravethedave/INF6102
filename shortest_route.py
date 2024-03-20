from utils2 import CustomRoute


def shortest_route(initial_route: CustomRoute) -> CustomRoute:
    """
    Hill climbing to find the best route for a truck knowing what pizzeria it visits
    Args:
        initial_route: Initial Route

    Returns: The result of hill climbing for a given path

    """
    shortest = initial_route
    continue_search = True
    while continue_search:
        continue_search = False
        neighbours = shortest.get_route_n()
        for n in neighbours:
            if n.distance_cost < shortest.distance_cost:
                shortest = n
                continue_search = True
                break
    return shortest
