import pandas as pd
from haversine import haversine
import heapq
import folium
import logging
from folium.plugins import MeasureControl
from folium.plugins import MiniMap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_excel('data/alagamentos.xlsx')
df_capitais = pd.read_csv('data/capitais.csv')

coords = []
idx = []

for index, row in df.iterrows():
        lat_str, lon_str = row['localidade'].split(',')
        coord = (float(lat_str), float(lon_str))
        
        if coord not in coords:
            coords.append(coord)
            idx.append(index)

df = df.loc[idx].reset_index(drop=True)

def calculate_distance(coord1, coord2):
    return haversine(coord1, coord2)

cities = []
for index, row in df.iterrows():
    lat_str, lon_str = row['localidade'].split(',')
    coord = (float(lat_str), float(lon_str))
    cities.append({'name': row['descricao'], 'coord': coord})

caps = {}
for index, row in df_capitais.iterrows():
    caps[row['descricao']] = (row['latitude'], row['longitude'])

class Node:
    def __init__(self, position_name, position_coord, visited_flood_cities_coords, 
                 g_score_tuple, path_taken, 
                 current_segment_km, num_recharges_taken):
        self.position_name = position_name
        self.position_coord = position_coord
        self.visited_flood_cities_coords = visited_flood_cities_coords
        
        self.g_score_tuple = g_score_tuple
        
        self.f_score_tuple = self.g_score_tuple 
        
        self.path_taken = path_taken
        self.current_segment_km = current_segment_km
        self.num_recharges_taken = num_recharges_taken

    def __lt__(self, other):
        return self.f_score_tuple < other.f_score_tuple

    def get_state_key(self):
        return (
            self.position_coord,
            self.visited_flood_cities_coords,
            round(self.current_segment_km, 2),
            self.num_recharges_taken
        )

def a_star_drone_planning(initial_cap, cities, caps):
  """
  A* algorithm for drone path planning to visit flood cities with fuel constraints.
  
  Args:
    start_capital_name_param: Name of the starting capital city
    all_flood_cities_list: List of flood cities to visit
    all_capitals_dict: Dictionary of all capital cities with coordinates
  
  Returns:
    Node: Best solution node found within iteration limit
  """
  # Validate starting capital
  start_capital_coord_param = caps.get(initial_cap)
  if not start_capital_coord_param:
    logging.error(f"Starting capital '{initial_cap}' not found in capitals dictionary.")
    return None

  # Initialize search structures
  priority_queue = []
  explored_states = {}
  
  # Define fuel capacity and initial state
  FUEL_CAPACITY_KM = 750
  MAX_ITERATIONS = 30000
  
  # Create initial node with optimized g-score structure
  initial_cost = (0, 0, 0.0)  # (negative_cities_visited, recharges_count, total_distance)
  
  initial_node = Node(
    position_name=initial_cap,
    position_coord=start_capital_coord_param,
    visited_flood_cities_coords=frozenset(),
    g_score_tuple=initial_cost,
    path_taken=[{
      'type': 'start_capital', 
      'name': initial_cap, 
      'coord': start_capital_coord_param, 
      'segment_km': 0, 
      'total_km': 0
    }],
    current_segment_km=0.0,
    num_recharges_taken=0
  )
  
  heapq.heappush(priority_queue, initial_node)
  optimal_solution = initial_node
  
  search_iterations = 0

  # Main A* search loop
  while priority_queue and search_iterations < MAX_ITERATIONS:
    search_iterations += 1
    
    # Get most promising node
    current_node = heapq.heappop(priority_queue)

    # Update best solution if current is better
    if current_node.g_score_tuple < optimal_solution.g_score_tuple:
      optimal_solution = current_node

    # Check if state already explored with better cost
    state_key = current_node.get_state_key()
    if state_key in explored_states:
      if explored_states[state_key] <= current_node.g_score_tuple:
        continue
    explored_states[state_key] = current_node.g_score_tuple

    # Generate successors: visit unvisited flood cities
    unvisited_cities = [city for city in cities 
               if city['coord'] not in current_node.visited_flood_cities_coords]
    
    for city_data in unvisited_cities:
      city_coordinate = city_data['coord']
      city_label = city_data['name']

      travel_distance = calculate_distance(current_node.position_coord, city_coordinate)
      
      # Check fuel constraint
      if current_node.current_segment_km + travel_distance <= FUEL_CAPACITY_KM:
        # Create new state after visiting city
        updated_visited_set = current_node.visited_flood_cities_coords | {city_coordinate}
        
        updated_cost = (
          -len(updated_visited_set),  # Maximize cities visited (negative for min-heap)
          current_node.num_recharges_taken,
          current_node.g_score_tuple[2] + travel_distance
        )
        
        extended_path = current_node.path_taken + [{
          'type': 'visit_city', 
          'name': city_label, 
          'coord': city_coordinate, 
          'segment_km': current_node.current_segment_km + travel_distance, 
          'total_km': updated_cost[2]
        }]
        
        city_visit_node = Node(
          position_name=city_label,
          position_coord=city_coordinate,
          visited_flood_cities_coords=updated_visited_set,
          g_score_tuple=updated_cost,
          path_taken=extended_path,
          current_segment_km=current_node.current_segment_km + travel_distance,
          num_recharges_taken=current_node.num_recharges_taken
        )
        
        heapq.heappush(priority_queue, city_visit_node)

    # Generate successors: recharge at capitals (if not at starting position without progress)
    can_recharge = (current_node.current_segment_km > 0 or 
             len(current_node.visited_flood_cities_coords) > 0 or 
             len(current_node.path_taken) == 1)
    
    if can_recharge:
      for capital_label, capital_coordinate in caps.items():
        recharge_distance = calculate_distance(current_node.position_coord, capital_coordinate)

        # Check if reachable with current fuel
        if current_node.current_segment_km + recharge_distance <= FUEL_CAPACITY_KM:
          recharge_cost = (
            current_node.g_score_tuple[0],  # Keep same cities visited count
            current_node.num_recharges_taken + 1,  # Increment recharge count
            current_node.g_score_tuple[2] + recharge_distance
          )
          
          recharge_path = current_node.path_taken + [{
            'type': 'recharge_at_capital', 
            'name': capital_label, 
            'coord': capital_coordinate, 
            'segment_km': current_node.current_segment_km + recharge_distance, 
            'total_km': recharge_cost[2]
          }]
          
          refuel_node = Node(
            position_name=capital_label,
            position_coord=capital_coordinate,
            visited_flood_cities_coords=current_node.visited_flood_cities_coords,
            g_score_tuple=recharge_cost,
            path_taken=recharge_path,
            current_segment_km=0.0,  # Reset fuel after recharge
            num_recharges_taken=current_node.num_recharges_taken + 1
          )
          
          heapq.heappush(priority_queue, refuel_node)

  logging.info(f"A* search completed after {search_iterations} iterations.")
  return optimal_solution

chosen_start_capital = 'Brasília'
if chosen_start_capital not in caps:
    logging.error(f"Capital {chosen_start_capital} not found in capitals list.")
    exit()

if chosen_start_capital and cities and caps:
    logging.info(f"Starting A* planning from {chosen_start_capital} with {len(cities)} flood cities and {len(caps)} capitals.")
    final_solution_node = a_star_drone_planning(chosen_start_capital, cities, caps)
else:
    logging.error("Invalid input data for A* planning.")

def visualize_route_on_map(solution_node, capitals, cities=None):
  """
  Creates interactive and detailed visualization of drone route with multiple layers and advanced controls
  """
  if not solution_node or not solution_node.path_taken:
    logging.error("No valid solution node or empty path found for visualization.")
    return None
  
  # Configure map centered on Brazil with multiple tiles
  brazil_center = [-15.7801, -47.9292]  # More precise coordinates of Brazil's center
  m = folium.Map(
    location=brazil_center, 
    zoom_start=5,
    tiles=None,
    control_scale=True,
    prefer_canvas=True
  )
  
  # Add different types of base maps
  folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
  
  # Improved colors and icons for different point types
  point_config = {
    'start_capital': {'color': 'darkgreen', 'icon': 'home', 'prefix': 'fa', 'size': 12},
    'visit_city': {'color': 'darkred', 'icon': 'exclamation-triangle', 'prefix': 'fa', 'size': 8}, 
    'recharge_at_capital': {'color': 'darkblue', 'icon': 'battery-full', 'prefix': 'fa', 'size': 10}
  }
  
  # Create layer groups for better organization
  main_route_group = folium.FeatureGroup(name='Rota Principal', show=True)
  refuel_group = folium.FeatureGroup(name='Rotas de Reabastecimento', show=True)
  visited_cities_group = folium.FeatureGroup(name='Cidades Monitoradas', show=True)
  reference_points_group = folium.FeatureGroup(name='Pontos de Referência', show=False)
  missed_cities_group = folium.FeatureGroup(name='Cidades Não Atendidas', show=True)
  
  # Store coordinates for different route types
  full_route_coordinates = []
  refuel_segments = []
  monitoring_segments = []
  
  # Process each route step with enriched information
  for step_index, step in enumerate(solution_node.path_taken):
    coordinate = step['coord']
    step_type = step['type']
    location_name = step['name']
    
    full_route_coordinates.append([coordinate[0], coordinate[1]])
    
    # Specific configuration for each point type
    config = point_config.get(step_type, {'color': 'gray', 'icon': 'info-sign', 'prefix': 'glyphicon', 'size': 8})
    
    # Calculate additional statistics for popup
    remaining_fuel = 750 - step.get('segment_km', 0)
    fuel_percentage = (remaining_fuel / 750) * 100
    
    flight_time_step = step.get('total_km', 0) / 100
    hours = int(flight_time_step)
    minutes = int((flight_time_step - hours) * 60)
    
    detailed_popup = f"""
    <div style="font-family: Arial; width: 300px;">
      <h4 style="margin: 0; color: {config['color']};">📍 Etapa {step_index + 1}</h4>
      <hr style="margin: 5px 0;">
      
      <b>🏷️ Tipo:</b> {step_type.replace('_', ' ').title()}<br>
      <b>📍 Local:</b> {location_name}<br>
      <b>🌍 Coordenadas:</b> {coordinate[0]:.4f}°, {coordinate[1]:.4f}°<br>
      
      <hr style="margin: 5px 0;">
      
      <b>⛽ Combustível no Segmento:</b> {step.get('segment_km', 0):.1f} km<br>
      <b>⛽ Combustível Restante:</b> {remaining_fuel:.1f} km ({fuel_percentage:.0f}%)<br>
      <b>🛣️ Distância Total Percorrida:</b> {step.get('total_km', 0):.1f} km<br>
      <b>⏱️ Tempo de Voo Acumulado:</b> {hours}h {minutes}min<br>
      
      <hr style="margin: 5px 0;">
      
      <small><i>Velocidade estimada: 100 km/h</i></small>
    </div>
    """
    
    # Add specific markers based on type
    if step_type == 'start_capital':
      start_marker = folium.Marker(
        location=[coordinate[0], coordinate[1]],
        popup=folium.Popup(detailed_popup, max_width=350),
        tooltip=f"🏠 ORIGEM: {location_name}",
        icon=folium.Icon(
          color=config['color'], 
          icon=config['icon'], 
          prefix=config['prefix'],
          icon_size=(15, 15)
        )
      )
      reference_points_group.add_child(start_marker)
      
    elif step_type == 'recharge_at_capital':
      recharge_marker = folium.Marker(
        location=[coordinate[0], coordinate[1]], 
        popup=folium.Popup(detailed_popup, max_width=350),
        tooltip=f"⚡ RECARGA: {location_name}",
        icon=folium.Icon(
          color=config['color'], 
          icon=config['icon'], 
          prefix=config['prefix'],
          icon_size=(12, 12)
        )
      )
      reference_points_group.add_child(recharge_marker)
      
    else:  # visit_city
      city_marker = folium.CircleMarker(
        location=[coordinate[0], coordinate[1]],
        radius=config['size'],
        popup=folium.Popup(detailed_popup, max_width=350),
        tooltip=f"🚨 MONITORADA: {location_name}",
        color=config['color'],
        fillColor=config['color'],
        fill=True,
        weight=3,
        fillOpacity=0.8
      )
      visited_cities_group.add_child(city_marker)
  
  # Create route lines with different styles and animations
  if len(full_route_coordinates) > 1:
    for i in range(len(solution_node.path_taken) - 1):
      current_step = solution_node.path_taken[i]
      next_step = solution_node.path_taken[i + 1]
      
      current_coord = [current_step['coord'][0], current_step['coord'][1]]
      next_coord = [next_step['coord'][0], next_step['coord'][1]]
      
      segment_distance = calculate_distance(current_step['coord'], next_step['coord'])
      segment_time = segment_distance / 80  # 80 km/h
      
      # Segment information
      segment_info = f"""
      <b>Segmento {i+1} → {i+2}</b><br>
      De: {current_step['name']}<br>
      Para: {next_step['name']}<br>
      Distância: {segment_distance:.1f} km<br>
      Tempo estimado: {segment_time:.1f}h
      """
      
      if next_step['type'] == 'recharge_at_capital':
        # Refuel route - wavy orange line
        refuel_line = folium.PolyLine(
          locations=[current_coord, next_coord],
          color='darkorange',
          weight=5,
          opacity=0.9,
          dash_array='15,10,5,10',
          popup=folium.Popup(f"⚡ Rota de Emergência<br>{segment_info}", max_width=200)
        )
        refuel_group.add_child(refuel_line)
        refuel_segments.append([current_coord, next_coord])
        
      elif next_step['type'] == 'visit_city':
        # Monitoring route - solid purple line with gradient
        monitoring_line = folium.PolyLine(
          locations=[current_coord, next_coord],
          color='darkviolet',
          weight=4,
          opacity=0.8,
          popup=folium.Popup(f"🚨 Rota de Monitoramento<br>{segment_info}", max_width=200)
        )
        main_route_group.add_child(monitoring_line)
        monitoring_segments.append([current_coord, next_coord])
        
      else:
        # Other movements - dotted gray line
        generic_line = folium.PolyLine(
          locations=[current_coord, next_coord],
          color='dimgray',
          weight=3,
          opacity=0.6,
          dash_array='5,5',
          popup=folium.Popup(f"➡️ Movimento<br>{segment_info}", max_width=200)
        )
        main_route_group.add_child(generic_line)
      
      # Add more elaborate distance indicators
      mid_lat = (current_coord[0] + next_coord[0]) / 2
      mid_lon = (current_coord[1] + next_coord[1]) / 2
      
      text_color = 'darkorange' if next_step['type'] == 'recharge_at_capital' else 'darkviolet'
      type_icon = '⚡' if next_step['type'] == 'recharge_at_capital' else '🚨'
      
      distance_indicator = folium.Marker(
        location=[mid_lat, mid_lon],
        icon=folium.DivIcon(
          html=f'''
          <div style="
            background: linear-gradient(45deg, white, #f0f0f0);
            border: 2px solid {text_color};
            border-radius: 8px;
            padding: 4px 8px;
            font-size: 11px;
            font-weight: bold;
            color: {text_color};
            text-align: center;
            white-space: nowrap;
            box-shadow: 2px 2px 4px rgba(0,0,0,0.4);
            transform: rotate(-5deg);
          ">{type_icon} {segment_distance:.0f}km</div>
          ''',
          icon_size=(60, 25),
          icon_anchor=(30, 12)
        )
      )
      main_route_group.add_child(distance_indicator)
  
  # Add all capitals as reference points
  for capital_name, capital_coord in capitals.items():
    capital_ref = folium.CircleMarker(
      location=[capital_coord[0], capital_coord[1]],
      radius=6,
      popup=f"🏛️ <b>Capital:</b> {capital_name}<br>📍 {capital_coord[0]:.4f}°, {capital_coord[1]:.4f}°",
      tooltip=f"🏛️ {capital_name}",
      color='lightsteelblue',
      fillColor='lightblue',
      fill=True,
      weight=2,
      opacity=0.7,
      fillOpacity=0.4
    )
    reference_points_group.add_child(capital_ref)
  
  # Process unvisited cities with detailed information
  if cities:
    visited_coordinates = set()
    for step in solution_node.path_taken:
      if step['type'] == 'visit_city':
        visited_coordinates.add(step['coord'])
    
    unvisited_cities_count = 0
    for city_info in cities:
      city_coord = city_info['coord']
      city_name = city_info['name']
      
      if city_coord not in visited_coordinates:
        unvisited_cities_count += 1
        
        # Calculate distance to nearest capital
        min_capital_distance = min(
          calculate_distance(city_coord, capital_coord) 
          for capital_coord in capitals.values()
        )
        nearest_capital = min(
          capitals.items(), 
          key=lambda x: calculate_distance(city_coord, x[1])
        )[0]
        
        unvisited_popup = f"""
        <div style="font-family: Arial;">
          <h4 style="color: gold; margin: 0;">⚠️ Cidade Não Atendida</h4>
          <hr style="margin: 5px 0;">
          <b>📍 Nome:</b> {city_name}<br>
          <b>🌍 Coordenadas:</b> {city_coord[0]:.4f}°, {city_coord[1]:.4f}°<br>
          <b>🏛️ Capital Mais Próxima:</b> {nearest_capital}<br>
          <b>📏 Distância até Capital:</b> {min_capital_distance:.1f} km<br>
          <hr style="margin: 5px 0;">
          <small><i>Requer planejamento futuro</i></small>
        </div>
        """
        
        unvisited_city = folium.CircleMarker(
          location=[city_coord[0], city_coord[1]],
          radius=7,
          popup=folium.Popup(unvisited_popup, max_width=280),
          tooltip=f"⚠️ NÃO ATENDIDA: {city_name}",
          color='gold',
          fillColor='yellow',
          fill=True,
          weight=2,
          opacity=0.9,
          fillOpacity=0.6
        )
        missed_cities_group.add_child(unvisited_city)
  
  # Add all groups to the map
  m.add_child(main_route_group)
  m.add_child(refuel_group)
  m.add_child(visited_cities_group)
  m.add_child(reference_points_group)
  m.add_child(missed_cities_group)
  
  # Calculate and display mission statistics
  visited_count = -solution_node.g_score_tuple[0]
  recharge_count = solution_node.g_score_tuple[1]
  total_distance = solution_node.g_score_tuple[2]
  total_hours = total_distance / 80
  total_cities = len(cities) if cities else visited_count
  success_rate = (visited_count / total_cities * 100) if total_cities > 0 else 100
  
  # Advanced statistics panel
  statistics_panel = f'''
  <div style="position: fixed; 
        top: 10px; right: 10px; width: 320px; height: auto; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        z-index: 9999; 
        font-size: 13px; 
        padding: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
  <h3 style="margin: 0 0 10px 0; text-align: center;">📊 Relatório da Missão</h3>
  <hr style="border-color: rgba(255,255,255,0.3);">
  
  <div style="display: flex; justify-content: space-between;">
    <div>
      <p><b>🎯 Taxa de Sucesso:</b> {success_rate:.1f}%</p>
      <p><b>🏙️ Cidades Visitadas:</b> {visited_count}/{total_cities}</p>
      <p><b>⚡ Reabastecimentos:</b> {recharge_count}</p>
    </div>
    <div>
      <p><b>🛣️ Distância Total:</b> {total_distance:.1f} km</p>
      <p><b>⏱️ Tempo de Voo:</b> {total_hours:.1f}h</p>
      <p><b>⛽ Eficiência:</b> {visited_count/(recharge_count+1):.1f} cidades/tanque</p>
    </div>
  </div>
  </div>
  '''
  m.get_root().html.add_child(folium.Element(statistics_panel))
  
  # Updated and more detailed legend
  detailed_legend = '''
  <div style="position: fixed; 
        bottom: 30px; left: 30px; width: 280px; height: auto; 
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #333;
        border-radius: 8px;
        z-index: 9999; 
        font-size: 12px; 
        padding: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
  <h4 style="margin: 0 0 8px 0; color: #333;">🗺️ Legenda</h4>
  <hr style="margin: 5px 0; border-color: #ddd;">
  
  <p style="margin: 3px 0;"><span style="color: darkgreen;">🏠</span> Capital de Origem</p>
  <p style="margin: 3px 0;"><span style="color: navy;">⚡</span> Pontos de Reabastecimento</p>
  <p style="margin: 3px 0;"><span style="color: darkred;">🚨</span> Cidades Monitoradas</p>
  <p style="margin: 3px 0;"><span style="color: gold;">⚠️</span> Cidades Não Atendidas</p>
  <p style="margin: 3px 0;"><span style="color: lightsteelblue;">🏛️</span> Outras Capitais</p>
  
  <hr style="margin: 5px 0; border-color: #ddd;">
  
  <p style="margin: 3px 0;"><span style="color: darkviolet; font-weight: bold;">━━━</span> Rota de Monitoramento</p>
  <p style="margin: 3px 0;"><span style="color: darkorange; font-weight: bold;">╌╌╌</span> Rota de Emergência</p>
  
  <hr style="margin: 5px 0; border-color: #ddd;">
  <small style="color: #666;"><i>💡 Clique nos elementos para mais informações</i></small>
  </div>
  '''
  m.get_root().html.add_child(folium.Element(detailed_legend))
  
  # Add distance measurement plugin
  m.add_child(MeasureControl())
  
  # Add minimap
  minimap = MiniMap(toggle_display=True, minimized=True)
  m.add_child(minimap)
  
  return m

if 'final_solution_node' in locals() and final_solution_node:
    mapa_rota = visualize_route_on_map(final_solution_node, caps, cities)
    if mapa_rota:
        mapa_rota.save('outputs/rota_drone.html')
        logging.info("Map created successfully at: rota_drone.html!")
else:
    logging.error("No valid solution node found to visualize on the map.")
    
def exibir_estatisticas_rota(solution_node, total_flood_cities=None):
    if not solution_node:
        logging.warning("No solution node provided for route statistics.")
        return
    
    num_visited = -solution_node.g_score_tuple[0] 
    num_recharges = solution_node.g_score_tuple[1]
    total_dist = solution_node.g_score_tuple[2]
    
    # Gerar relatório detalhado e salvar em arquivo
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    relatorio_filename = f"outputs/relatorio_rota_drone_{timestamp}.txt"
    
    with open(relatorio_filename, 'w', encoding='utf-8') as arquivo:
      arquivo.write("=" * 80 + "\n")
      arquivo.write("RELATÓRIO DETALHADO DA MISSÃO DO DRONE\n")
      arquivo.write("=" * 80 + "\n")
      arquivo.write(f"Data/Hora da Geração: {pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
      arquivo.write(f"Capital de Partida: {chosen_start_capital}\n")
      arquivo.write(f"Autonomia do Drone: 750 km por segmento\n\n")
      
      # Resumo executivo
      arquivo.write("RESUMO EXECUTIVO:\n")
      arquivo.write("-" * 50 + "\n")
      if total_flood_cities:
        eficiencia = (num_visited/total_flood_cities*100)
        arquivo.write(f"✓ Taxa de Cobertura: {num_visited}/{total_flood_cities} cidades ({eficiencia:.1f}%)\n")
      arquivo.write(f"✓ Reabastecimentos Necessários: {num_recharges}\n")
      arquivo.write(f"✓ Distância Total da Missão: {total_dist:.2f} km\n")
      arquivo.write(f"✓ Tempo Estimado de Voo: {total_dist/100:.1f} horas\n\n")
      
      # Análise estatística adicional
      arquivo.write("ANÁLISE ESTATÍSTICA COMPLEMENTAR:\n")
      arquivo.write("-" * 50 + "\n")
      
      distancias_entre_pontos = []
      for i in range(1, len(solution_node.path_taken)):
        coord_anterior = solution_node.path_taken[i-1]['coord']
        coord_atual = solution_node.path_taken[i]['coord']
        dist = calculate_distance(coord_anterior, coord_atual)
        distancias_entre_pontos.append(dist)
      
      if distancias_entre_pontos:
        arquivo.write(f"• Distância média entre pontos: {sum(distancias_entre_pontos)/len(distancias_entre_pontos):.1f} km\n")
        arquivo.write(f"• Menor distância percorrida: {min(distancias_entre_pontos):.1f} km\n")
        arquivo.write(f"• Maior distância percorrida: {max(distancias_entre_pontos):.1f} km\n")
      
      arquivo.write(f"• Eficiência energética: {num_visited/(num_recharges+1):.1f} cidades por tanque\n")
      arquivo.write(f"• Consumo médio: {total_dist/num_visited:.1f} km por cidade visitada\n")
      arquivo.write(f"• Distância média por reabastecimento: {total_dist/(num_recharges+1):.1f} km\n")
      arquivo.write(f"• Média de reabastecimentos por cidade visitada: {num_recharges/num_visited:.2f}\n")
      
      # Roteiro cronológico completo
      arquivo.write("\nROTEIRO CRONOLÓGICO DA MISSÃO:\n")
      arquivo.write("-" * 50 + "\n")
      
      for ordem, movimento in enumerate(solution_node.path_taken, 1):
        tipo_operacao = {
          'start_capital': 'DECOLAGEM',
          'visit_city': 'MONITORAMENTO', 
          'recharge_at_capital': 'REABASTECIMENTO'
        }
        
        simbolo = {
          'start_capital': '🚁',
          'visit_city': '📍', 
          'recharge_at_capital': '⛽'
        }
        
        operacao = tipo_operacao.get(movimento['type'], 'MOVIMENTO')
        icone = simbolo.get(movimento['type'], '•')
        
        arquivo.write(f"{ordem:2d}. {icone} {operacao}: {movimento['name']}\n")
        arquivo.write(f"    Localização: {movimento['coord'][0]:.4f}°, {movimento['coord'][1]:.4f}°\n")
        
        if ordem > 1:  # Não mostrar distância para o ponto inicial
          distancia_etapa = movimento.get('segment_km', 0) - solution_node.path_taken[ordem-2].get('segment_km', 0)
          if movimento['type'] == 'recharge_at_capital' and ordem > 1:
            # Calcular distância para reabastecimento
            coord_anterior = solution_node.path_taken[ordem-2]['coord']
            distancia_etapa = calculate_distance(coord_anterior, movimento['coord'])
          arquivo.write(f"    Distância da etapa anterior: {distancia_etapa:.1f} km\n")
        
        arquivo.write(f"    Distância acumulada: {movimento.get('total_km', 0):.1f} km\n")
        arquivo.write(f"    Combustível no segmento: {movimento.get('segment_km', 0):.1f} km\n\n")
      
      
      
      arquivo.write("\n" + "=" * 80 + "\n")
      arquivo.write("FIM DO RELATÓRIO\n")
      arquivo.write("=" * 80 + "\n")
    
    logging.info(f"Relatório detalhado salvo em: {relatorio_filename}")
    print(f"\n📄 Relatório completo da missão salvo em: {relatorio_filename}")

if 'final_solution_node' in locals() and final_solution_node:
    exibir_estatisticas_rota(final_solution_node, len(cities))
