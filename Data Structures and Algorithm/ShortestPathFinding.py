import heapq
import osmnx as ox #find graph edge travel time. download geospatila data from OpenStreetMap and model, project, visualize and analysie real-world street networks
import networkx as nx #used for finding routes between 2 points 
from geopy.geocoders import Nominatim #geopy is used to find coordinates(langitude, longtitude) given the location names

INSTRUCTIONS = f"""
    {'=' * 50} DIJKSTRA'S ALGORITHM {'=' * 51}
    |  Instructions Manual                                                                                                    | 
    |  1) Input the details of your search.                                                                                   | 
    |      i)   search boundary: town, city, state, territory, country                                                        | 
    |      ii)  source location: address of your source                                                                       | 
    |      iii) destination location: address of your destination                                                             | 
    |      iv)  mode of transportation: how are you travelling?                                                               | 
    |      v)   optimiser: what would you like to prioritise, time or distance?                                               | 
    |   2) Wait for the program to run(this will take about 1-2mins)                                                          | 
    |   3) When the program ends, as indicated by (END OF PROGRAM), in your working directory a map.html file is generated.   | 
    |      To view your map, run the file                                                                          |           
    {'=' * 122}
"""

#node to represent each node on the graph
class Node: 
    def __init__(self, name, wsf = 0):
        self.name = name
        self.adjacentNodes = {} #stores pathcost to adjacent nodes
        self.wsf = wsf #calculates the path cost from the source node
    
    def addAdjacentNodes(self, node, pathCost):
        self.adjacentNodes[node] = pathCost

    def __lt__(self, node2): #heap will organise the queue based on wsf
        return self.wsf < node2.wsf
    
#function to get the user input and print out instructions
def get_user_input():
    G_name = input("Boundary of search(e.g., 'Subang Jaya, Malaysia'): ")
    source_node_name = input("Source location(e.g., 'Sunway University'): ")
    destination_node_name = input("Destination location(e.g., 'Sunway Pyramid'): ")
    mode = input("Travel mode('drive', 'bike', 'walk'): ")
    optimiser = input("Optimiser('length', 'travel_time'): ")
    print("")  
    return G_name, source_node_name, destination_node_name, mode, optimiser

#accepts strings and returns the networkx.MultiDiGraph, sourceID and destinationID   
def get_nodeID(G_name, source_node_name, destination_node_name, mode):
    
    G =  ox.graph_from_place(G_name, network_type = mode)
    #to add time traveled
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)     

    #use geocoder to get latitude and longitude of source and destination nodes
    geolocator = Nominatim(user_agent="my_request")
    source = geolocator.geocode(source_node_name)
    destination= geolocator.geocode(destination_node_name)

    #find the ids of the source and destination
    source_id = ox.distance.nearest_nodes(G, [source.longitude], [source.latitude])[0]
    destination_id = ox.distance.nearest_nodes(G, [destination.longitude], [destination.latitude])[0]

    return G, source_id, destination_id

#accepts input of networkx.MultiDiGraph and optimiser(string) and returns a list of (sourceID, neighbourID, pathcost)
def get_state_space(G, optimiser):
    state_space = []
    no_of_edges = 0
    #for all the edges in the map, create state space of [sourceID, destinationID, pathcost]
    for edge in G.out_edges(data=True):
        no_of_edges += 1
        edge_attributes = edge[2]
        # remove geometry object from output
        edge_attributes_wo_geometry = {i:edge_attributes[i] for i in edge_attributes if i!='geometry'}
        state_space.append([edge[0], edge[1],edge_attributes_wo_geometry[optimiser]])
    return state_space

#accepts a list of [sourceID, nodeID, pathcost] and returns a map that maps nodeID to the nodeObject 
def create_graph_from_list(state_space):
    node_map = {} #list of nodes
    for source_id, destination_id, path_cost in state_space:
        if source_id not in node_map:
            node_map[source_id] = Node(source_id)
        if destination_id not in node_map:
            node_map[destination_id] = Node(destination_id)
        node_map[source_id].addAdjacentNodes(node_map[destination_id], path_cost)
    return node_map

#takes in user's input and process it for suitable use and call the Dijkstra Algorithm, this returns the networkx.MultiDiGraph and the shortest path calculated by Dijkstra's
def run_djikstra(G_name, source_node_name, destination_node_name, mode, optimiser):
    G, source_id, destination_id = get_nodeID(G_name, source_node_name, destination_node_name, mode)
    #from the map, generate the state space of all connecting nodes on the given map
    state_space = get_state_space(G, optimiser)
    #initialise all nodes and link them 
    node_map = create_graph_from_list(state_space)
    shortest_path = dijkstra(node_map[source_id], node_map[destination_id])
    return G, shortest_path

#Dijkstra algorithm
def dijkstra(source, destination):
    #stores cheapest known price from starting to all other known destinations(uses hashtables)
    cheapest_prices_table = {}
    #stores cheapeast previous stopover city to reach the cheapest price (hashtable)
    cheapest_previous_stopover_node = {}
    #list of visited nodes 
    visited = {}
    #list of unvisited nodes
    unvisited = []
    heapq.heapify(unvisited)
    
    #initialise starting city
    current_node = source
    cheapest_prices_table[source] = 0
    heapq.heappush(unvisited, source)
    
    #while there are still unvisited nodes
    while unvisited:
        current_node = unvisited.pop(0)
        visited[current_node] = True
        #for each adjacent city of the current node
        for adjacentNode, pathCost in current_node.adjacentNodes.items():
            pathCost_through_current_node = cheapest_prices_table[current_node] + pathCost
            #if the adjacent node is a newly discovered node or we found a cheaper pathcost to the adjavent node
            if (adjacentNode not in cheapest_prices_table) or (pathCost_through_current_node < cheapest_prices_table[adjacentNode]):
                cheapest_prices_table[adjacentNode] = pathCost_through_current_node
                cheapest_previous_stopover_node[adjacentNode] = current_node
                adjacentNode.wsf = pathCost_through_current_node #to keep track of the adjacent node's pathcost from the source
                # if it is not in visited, then add to the list of unvisited nodes
                if adjacentNode not in visited and adjacentNode not in unvisited:
                    heapq.heappush(unvisited, adjacentNode)

    #finding the shortest path by backtracking the cheapest_previous_stopover_node
    shortest_path = [source]
    pointer = destination
    while pointer != source:
        shortest_path.insert(1, pointer)
        pointer = cheapest_previous_stopover_node[pointer]
    shortest_path_name = []
    for i in shortest_path:
        shortest_path_name.append(i.name)

    return shortest_path_name

#function that prints out total time and distance travelled and save map into html file
def display_output(G, shortest_path):
    shortest_path_edges = ox.utils_graph.route_to_gdf(G, shortest_path)
    km = sum(shortest_path_edges['length'])/1000
    minutes = sum(shortest_path_edges['travel_time'])/60
    print(f'''    {'=' * 41}
    |  Total kilometers of path(km): {km: .3}  |
    |  Total travel time(minutes): {minutes: .3}{' '*4}|
    {'=' * 41}

Open map.html to view your route
- END OF PROGRAM -
''')
   
    m = shortest_path_edges.explore(style_kwds={'weight': 6}, marker_type = 'circle', marker_kwds={'radius': 100})
    m.save('map.html')


#change to suit user input
def main():
    print(INSTRUCTIONS)
    #get user input
    G_name, source_node_name, destination_node_name, mode, optimiser = get_user_input()
    #run djikstra 
    G, shortest_path = run_djikstra(G_name, source_node_name, destination_node_name, mode, optimiser)
    #display map
    display_output(G, shortest_path)

if __name__ == "__main__":
    main()


