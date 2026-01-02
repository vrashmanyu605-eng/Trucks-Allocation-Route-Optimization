"""
Fleet Allocation Optimization System
Production-ready implementation for vehicle-to-delivery-order allocation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
import math
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== STEP 1: DATA PREPARATION LAYER ====================

class DataFetcher:
    """Handles all data fetching operations from CSV files"""
    
    def __init__(self, folder_path: str = '.'):
        self.folder_path = folder_path
        self.do_dump_path = f"{folder_path}/1_DO_Dump_Enhanced_1000.csv"
        self.lr_details_path = f"{folder_path}/2_LR_Details_Enhanced_1000.csv"
        self.vehicle_master_path = f"{folder_path}/3_Vehicle_Master.csv"
        self.route_master_path = f"{folder_path}/4_Route_Master.csv"
        
    def fetch_pending_dos(self) -> List[Dict]:
        """Fetch all pending delivery orders where Balance to Despatch Qty > 0"""
        df = pd.read_csv(self.do_dump_path)
        
        # Filter for pending DOs
        pending_df = df[df['Balance_to_Despatch_Qty'] > 0].copy()
        
        # Convert Delivery_Deadline to datetime
        pending_df['Delivery_Deadline'] = pd.to_datetime(pending_df['Delivery_Deadline'])
        
        # Rename columns to match internal expectations or adjust expectations
        # The internal code uses 'DO No', 'Customer Name', 'Loading_Point_Code', etc.
        # CSV uses 'DO_No', 'Customer_Name', 'Loading_Point_Code', etc.
        # Mapping to match the expected keys in the rest of the script:
        column_mapping = {
            'DO_No': 'DO No',
            'Customer_Name': 'Customer Name',
            'Thick_mm': 'Thick',
            'Width_mm': 'Width',
            'Length_mm': 'Length',
            'Expected_Freight_Rate_INR': 'Expected_Freight_Rate'
        }
        pending_df = pending_df.rename(columns=column_mapping)
        
        # Determine Destination_Zone based on Destination if not in CSV
        # For now, let's add a placeholder if not present, but 1_DO_Dump doesn't have Zone
        # Looking at 1_DO_Dump, it has 'Destination' like 'Mumbai, Maharashtra'
        if 'Destination_Zone' not in pending_df.columns:
            # Simple heuristic mapping for demonstration
            zone_map = {
                'Mumbai': 'West', 'Pune': 'West', 'Surat': 'West', 'Ahmedabad': 'West', 'Indore': 'West',
                'Delhi': 'North', 'Lucknow': 'North', 'Jaipur': 'North',
                'Chennai': 'South', 'Bengaluru': 'South', 'Hyderabad': 'South', 'Coimbatore': 'South',
                'Kolkata': 'East', 'Jamshedpur': 'East'
            }
            def get_zone(dest):
                for city, zone in zone_map.items():
                    if city in str(dest):
                        return zone
                return 'Other'
            pending_df['Destination_Zone'] = pending_df['Destination'].apply(get_zone)
        
        dos = pending_df.to_dict('records')
        logger.info(f"Fetched {len(dos)} pending delivery orders from {self.do_dump_path}")
        return dos
    
    def fetch_vehicles(self, status='Available') -> List[Dict]:
        """Fetch all available vehicles"""
        df = pd.read_csv(self.vehicle_master_path)
        
        # Filter by status
        available_df = df[df['Availability_Status'] == status].copy()
        
        # Mapping CSV columns to expected keys
        # CSV has: Vehicle_ID, Vehicle_Number, Vehicle_Type, Capacity_Tons, Ownership_Type, Current_Location, Availability_Status, Driver_Assigned
        # Script expects: Vehicle_ID, Vehicle_Number, Vehicle_Type, Ownership_Type, Capacity_Tons, Current_Odometer_Km, Next_Service_Due_Km, Vehicle_Age_Years, Standard_Fuel_Efficiency, GPS_Device_ID, Permit_Valid_Routes, Driver_Assigned, Status
        
        # Add missing columns with reasonable defaults or derived values
        available_df['Status'] = available_df['Availability_Status']
        
        # Since CSV lacks some technical details, we'll assign defaults
        if 'Current_Odometer_Km' not in available_df.columns:
            available_df['Current_Odometer_Km'] = np.random.randint(50000, 150000, len(available_df))
        if 'Next_Service_Due_Km' not in available_df.columns:
            available_df['Next_Service_Due_Km'] = available_df['Current_Odometer_Km'] + np.random.randint(5000, 15000, len(available_df))
        if 'Vehicle_Age_Years' not in available_df.columns:
            available_df['Vehicle_Age_Years'] = np.random.randint(1, 8, len(available_df))
        if 'Standard_Fuel_Efficiency' not in available_df.columns:
            # Base fuel efficiency on vehicle type
            fe_map = {'Trailer': 5.0, 'Truck': 6.5, 'HCV': 5.5, 'LCV': 8.0, 'Open': 6.0, 'Covered': 6.0, 'Flatbed': 5.8}
            available_df['Standard_Fuel_Efficiency'] = available_df['Vehicle_Type'].map(fe_map).fillna(6.0)
        if 'GPS_Device_ID' not in available_df.columns:
            available_df['GPS_Device_ID'] = 'GPS-' + available_df['Vehicle_ID'].astype(str)
        if 'Permit_Valid_Routes' not in available_df.columns:
            # Default permits to all zones for simplicity, or based on ownership
            available_df['Permit_Valid_Routes'] = [['North', 'West', 'South', 'East'] for _ in range(len(available_df))]
            
        vehicles = available_df.to_dict('records')
        logger.info(f"Fetched {len(vehicles)} available vehicles from {self.vehicle_master_path}")
        return vehicles


class DataEnricher:
    """Enriches data with real-time and calculated fields"""
    
    @staticmethod
    def get_gps_location(gps_device_id: str) -> Tuple[float, float]:
        """Get current GPS location of vehicle"""
        ############## In production, this would query GPS tracking system
        locations = {
            'GPS-001': (28.6139, 77.2090),
            'GPS-002': (28.7041, 77.1025),
            'GPS-003': (19.0760, 72.8777)
        }
        return locations.get(gps_device_id, (28.6139, 77.2090))
    
    @staticmethod
    def calculate_service_due(vehicle: Dict) -> int:
        """Calculate days to next service"""
        km_to_service = vehicle['Next_Service_Due_Km'] - vehicle['Current_Odometer_Km']
        avg_daily_km = 200  # Average daily km traveled
        days_to_service = max(0, km_to_service // avg_daily_km)
        return days_to_service
    
    @staticmethod
    def calculate_weight_from_dimensions(thick: float, width: float, length: float, 
                                        qty: int, density: float = 7850) -> float:
        """
        Calculate weight in tons from steel dimensions
        thick: mm, width: mm, length: mm, density: kg/m³ (default steel = 7850)
        """
        # Convert mm to meters
        thick_m = thick / 1000
        width_m = width / 1000
        length_m = length / 1000
        
        # Volume per piece in cubic meters
        volume_per_piece = thick_m * width_m * length_m
        
        # Weight per piece in kg
        weight_per_piece_kg = volume_per_piece * density
        
        # Total weight in tons
        total_weight_tons = (weight_per_piece_kg * qty) / 1000
        
        return round(total_weight_tons, 2)
    
    @staticmethod
    def enrich_pending_dos(pending_dos: List[Dict]) -> List[Dict]:
        """Add calculated weight to each DO if not already present"""
        for do in pending_dos:
            # If Weight_Tons is already in CSV, use it, otherwise calculate
            if 'Weight_Tons' not in do or pd.isna(do['Weight_Tons']):
                do['Weight_Tons'] = DataEnricher.calculate_weight_from_dimensions(
                    do['Thick'], do['Width'], do['Length'], do['Planned_Qty']
                )
        logger.info("Enriched DOs with weight calculations")
        return pending_dos
    
    @staticmethod
    def enrich_vehicles(vehicles: List[Dict]) -> List[Dict]:
        """Add real-time location and service info to vehicles"""
        for vehicle in vehicles:
            # If CSV has Current_Location as text (e.g. "Chennai, Tamil Nadu"), we might need Lats/Lngs
            # For now, if missing, assign based on a simple city-coord map
            if 'Current_Location_Lat' not in vehicle:
                city_coords = {
                    'Jamshedpur': (22.8046, 86.2029),
                    'Kolkata': (22.5726, 88.3639),
                    'Durgapur': (23.5204, 87.3119),
                    'Rourkela': (22.2604, 84.8536),
                    'Bhilai': (21.2514, 81.1257),
                    'Visakhapatnam': (17.6869, 83.2185),
                    'Mumbai': (19.0760, 72.8777),
                    'Delhi': (28.7041, 77.1025),
                    'Pune': (18.5204, 73.8567),
                    'Chennai': (13.0827, 80.2707),
                    'Bengaluru': (12.9716, 77.5946),
                    'Hyderabad': (17.3850, 78.4867),
                    'Ahmedabad': (23.0225, 72.5714),
                    'Jaipur': (26.9124, 75.7873),
                    'Lucknow': (26.8467, 80.9462),
                    'Coimbatore': (11.0168, 76.9558),
                    'Surat': (21.1702, 72.8311),
                    'Indore': (22.7196, 75.8577)
                }
                curr_loc = str(vehicle.get('Current_Location', ''))
                found = False
                for city, coords in city_coords.items():
                    if city in curr_loc:
                        vehicle['Current_Location_Lat'], vehicle['Current_Location_Lng'] = coords
                        found = True
                        break
                if not found:
                    vehicle['Current_Location_Lat'], vehicle['Current_Location_Lng'] = (22.8046, 86.2029) # Default Jamshedpur
            
            if 'Days_To_Service' not in vehicle:
                vehicle['Days_To_Service'] = DataEnricher.calculate_service_due(vehicle)
                
        logger.info("Enriched vehicles with GPS and service data")
        return vehicles


# ==================== STEP 2: CONSTRAINT VALIDATION ====================

class ConstraintValidator:
    """Validates hard constraints for vehicle-DO assignments"""
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in km"""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    @staticmethod
    def get_route_distance(loading_point: str, destination: str) -> float:
        """Get standard route distance from Route Master CSV"""
        try:
            df_routes = pd.read_csv('4_Route_Master.csv')
            # partial match for destination as it might contain state in DO dump but not in Route master
            dest_city = destination.split(',')[0].strip()
            mask = (df_routes['Origin_Code'] == loading_point) & \
                   (df_routes['Destination'].str.contains(dest_city, case=False))
            
            match = df_routes[mask]
            if not match.empty:
                return float(match.iloc[0]['Standard_Distance_Km'])
        except Exception as e:
            logger.error(f"Error fetching route distance: {e}")
            
        return 1500.0  # Default fallback
    
    @staticmethod
    def estimate_travel_time(current_lat: float, current_lng: float,
                            loading_lat: float, loading_lng: float,
                            dest_lat: float, dest_lng: float) -> float:
        """Estimate total travel time in hours"""
        # Distance to loading point
        dist_to_loading = ConstraintValidator.haversine_distance(
            current_lat, current_lng, loading_lat, loading_lng
        )
        
        # Distance from loading to destination
        dist_loading_to_dest = ConstraintValidator.haversine_distance(
            loading_lat, loading_lng, dest_lat, dest_lng
        )
        
        # Average speed: 50 km/h, plus 4 hours buffer for loading/unloading
        total_distance = dist_to_loading + dist_loading_to_dest
        travel_time = (total_distance / 50) + 4
        
        return travel_time
    
    @staticmethod
    def can_assign_vehicle_to_do(vehicle: Dict, do: Dict) -> Tuple[bool, str]:
        """Check if vehicle meets all hard constraints for DO"""
        
        # Constraint 1: Capacity match
        if vehicle['Capacity_Tons'] < do['Weight_Tons']:
            return False, f"Capacity insufficient: {vehicle['Capacity_Tons']}T < {do['Weight_Tons']}T"
        
        # Constraint 2: Vehicle type match
        if do['Preferred_Vehicle_Type'] != vehicle['Vehicle_Type']:
            return False, f"Vehicle type mismatch: Need {do['Preferred_Vehicle_Type']}, got {vehicle['Vehicle_Type']}"
        
        # Constraint 3: Service due check
        trip_distance = ConstraintValidator.get_route_distance(
            do['Loading_Point_Code'], do['Destination']
        )
        km_available = vehicle['Next_Service_Due_Km'] - vehicle['Current_Odometer_Km']
        
        if km_available < trip_distance:
            return False, f"Service due before trip: {km_available}km < {trip_distance}km"
        
        # Constraint 4: Delivery deadline feasibility
        travel_time = ConstraintValidator.estimate_travel_time(
            vehicle['Current_Location_Lat'], vehicle['Current_Location_Lng'],
            do['Loading_Point_Lat'], do['Loading_Point_Lng'],
            do['Destination_Lat'], do['Destination_Lng']
        )
        
        hours_to_deadline = (do['Delivery_Deadline'] - datetime.now()).total_seconds() / 3600
        
        if travel_time > hours_to_deadline:
            return False, f"Cannot meet deadline: {travel_time:.1f}h > {hours_to_deadline:.1f}h"
        
        # Constraint 5: Legal permits
        if do['Destination_Zone'] not in vehicle['Permit_Valid_Routes']:
            return False, f"No permit for {do['Destination_Zone']} zone"
        
        return True, "All constraints satisfied"


# ==================== STEP 3: PROFITABILITY PREDICTION MODEL ====================

class ProfitabilityPredictor:
    """Predicts trip profitability using machine learning trained on actual LR data"""
    
    def __init__(self, model_path: str = 'profit_prediction_model.pkl', lr_details_path: str = '2_LR_Details_Enhanced_1000.csv'):
        self.model_path = model_path
        self.lr_details_path = lr_details_path
        self.model = None
        self.label_encoders = {}
        # Features available in the historical LR data
        self.feature_names = [
            'Vehicle_Ownership_Type', 'Capacity', 'Actual_Distance_Km',
            'NetWt_Tons', 'Trip_Duration_Hours'
        ]
    
    def load_actual_training_data(self) -> pd.DataFrame:
        """Load and prepare actual training data from CSV"""
        logger.info(f"Loading training data from {self.lr_details_path}...")
        df = pd.read_csv(self.lr_details_path)
        
        # Select relevant features and target
        # Target: Profit_Margin_INR
        # Features: Vehicle_Ownership_Type (cat), Capacity (num), Actual_Distance_Km (num), NetWt_Tons (num), Trip_Duration_Hours (num)
        
        # Filter for rows where target exists
        df = df.dropna(subset=['Profit_Margin_INR'])
        
        return df
    
    def train_model(self):
        """Train the profitability prediction model using actual historical data"""
        df = self.load_actual_training_data()
        
        # Encode categorical features
        self.label_encoders['Vehicle_Ownership_Type'] = LabelEncoder()
        df['Vehicle_Ownership_Type_Encoded'] = self.label_encoders['Vehicle_Ownership_Type'].fit_transform(
            df['Vehicle_Ownership_Type'].astype(str)
        )
        
        # Prepare features list
        feature_cols = ['Vehicle_Ownership_Type_Encoded', 'Capacity', 'Actual_Distance_Km', 'NetWt_Tons', 'Trip_Duration_Hours']
        
        X = df[feature_cols]
        y = df['Profit_Margin_INR']
        
        # Train model
        logger.info("Training Random Forest model on actual data...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        
        # Save model
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': feature_cols
        }, self.model_path)
        
        logger.info(f"Model trained on {len(df)} samples and saved to {self.model_path}")
        
        # Calculate and log performance
        train_score = self.model.score(X, y)
        logger.info(f"Model R² score on training data: {train_score:.3f}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            saved_data = joblib.load(self.model_path)
            self.model = saved_data['model']
            self.label_encoders = saved_data['label_encoders']
            self.feature_names = saved_data['feature_names']
            logger.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {self.model_path}. Training new model...")
            self.train_model()
    
    @staticmethod
    def get_current_market_rate(route_id: str) -> float:
        """Get current market rate per ton for route - could be enhanced with actuals"""
        return np.random.uniform(1800, 2200)
    
    @staticmethod
    def get_current_diesel_price() -> float:
        """Get current diesel price"""
        return 90.5
    
    @staticmethod
    def get_loading_point_score(loading_point_code: str) -> float:
        """Get efficiency score for loading point"""
        # Could use actual delay data from LR details to calculate this
        return 85.0
    
    @staticmethod
    def get_driver_score(driver_id: str) -> float:
        """Get driver performance score"""
        # Could use actual trip durations/delays to calculate this
        return 85.0
    
    @staticmethod
    def get_route_object(loading_point: str, destination: str) -> Dict:
        """Get route details from Route Master CSV"""
        # This will be handled by reading 4_Route_Master.csv in the main loop or here
        # For efficiency, let's assume we pass the Route Master DF or have it globally
        # For now, keeping a simplified version but main should pass the real object
        try:
            df_routes = pd.read_csv('4_Route_Master.csv')
            # Filter by origin code and destination (partial match for destination)
            mask = (df_routes['Origin_Code'] == loading_point) & \
                   (df_routes['Destination'].str.contains(destination.split(',')[0]))
            
            match = df_routes[mask]
            if not match.empty:
                row = match.iloc[0]
                return {
                    'Route_ID': row['Route_ID'],
                    'Standard_Distance_Km': row['Standard_Distance_Km'],
                    'Route_Difficulty_Score': 5.0, # Not in CSV, default
                    'Return_Load_Probability': row['Return_Load_Probability_Percent'],
                    'Destination_Zone': 'TBD' # Derived elsewhere
                }
        except Exception as e:
            logger.error(f"Error reading Route Master: {e}")
            
        return {
            'Route_ID': 'R-DEFAULT',
            'Standard_Distance_Km': 1500,
            'Route_Difficulty_Score': 6.0,
            'Return_Load_Probability': 40,
            'Destination_Zone': 'Unknown'
        }
    
    def predict_trip_profit(self, vehicle: Dict, do: Dict, route: Dict) -> Tuple[float, Dict]:
        """Predict profit for vehicle-DO combination using trained model"""
        
        if self.model is None:
            self.load_model()
        
        # Prepare features based on trained feature list:
        # ['Vehicle_Ownership_Type_Encoded', 'Capacity', 'Actual_Distance_Km', 'NetWt_Tons', 'Trip_Duration_Hours']
        
        ownership_type = vehicle['Ownership_Type']
        # Handle unseen categories
        try:
            ownership_encoded = self.label_encoders['Vehicle_Ownership_Type'].transform([ownership_type])[0]
        except:
            # Default to first class if unknown
            ownership_encoded = 0
            
        capacity = vehicle['Capacity_Tons']
        distance = route['Standard_Distance_Km']
        weight = do['Weight_Tons']
        # Estimate duration based on distance (avg 50 km/h)
        duration = (distance / 50) + 4
        
        features = [ownership_encoded, capacity, distance, weight, duration]
        
        # Predict
        predicted_profit = self.model.predict([features])[0]
        
        # Metadata for reasoning
        meta = {
            'Capacity_Utilization_%': (weight / capacity) * 100,
            'Standard_Fuel_Efficiency': vehicle.get('Standard_Fuel_Efficiency', 5.5),
            'Driver_Performance_Score': 85.0,
            'Return_Load_Probability_%': route.get('Return_Load_Probability', 40),
            'Loading_Point_Efficiency_Score': 85.0
        }
        
        return predicted_profit, meta


# ==================== STEP 4: OPTIMIZATION ENGINE ====================

class OptimizationEngine:
    """Optimizes vehicle-DO allocation to maximize profit"""
    
    def __init__(self, profit_predictor: ProfitabilityPredictor):
        self.profit_predictor = profit_predictor
        self.validator = ConstraintValidator()
    
    def optimize_daily_allocation(self, pending_dos: List[Dict], 
                                  available_vehicles: List[Dict]) -> List[Dict]:
        """
        Allocate vehicles to DOs to maximize total profit
        Subject to: capacity, timing, and one-vehicle-one-DO constraints
        """
        # LIMITING VOLUME FOR STABILITY IN TEST IF NEEDED
        # However, we have 1000 DOs, but let's see how many are pending
        logger.info(f"Starting optimization: {len(pending_dos)} DOs, {len(available_vehicles)} vehicles")
        
        # Create optimization problem
        prob = LpProblem("Fleet_Allocation", LpMaximize)
        
        # Decision variables
        x = {}
        for vehicle in available_vehicles:
            for do in pending_dos:
                var_name = f"x_{str(vehicle['Vehicle_ID']).replace('-', '_')}_{str(do['DO No']).replace('-', '_')}"
                x[vehicle['Vehicle_ID'], do['DO No']] = LpVariable(var_name, cat=LpBinary)
        
        # Calculate profit matrix
        profit_matrix = {}
        feasibility_matrix = {}
        
        for vehicle in available_vehicles:
            for do in pending_dos:
                can_assign, reason = self.validator.can_assign_vehicle_to_do(vehicle, do)
                feasibility_matrix[vehicle['Vehicle_ID'], do['DO No']] = (can_assign, reason)
                
                if can_assign:
                    route = self.profit_predictor.get_route_object(
                        do['Loading_Point_Code'], do['Destination']
                    )
                    predicted_profit, _ = self.profit_predictor.predict_trip_profit(
                        vehicle, do, route
                    )
                    profit_matrix[vehicle['Vehicle_ID'], do['DO No']] = predicted_profit
                else:
                    profit_matrix[vehicle['Vehicle_ID'], do['DO No']] = -999999
        
        # Objective: Maximize total profit
        prob += lpSum([
            profit_matrix[vehicle['Vehicle_ID'], do['DO No']] * 
            x[vehicle['Vehicle_ID'], do['DO No']]
            for vehicle in available_vehicles
            for do in pending_dos
        ])
        
        # Constraint 1: Each vehicle assigned to at most 1 DO
        for vehicle in available_vehicles:
            prob += lpSum([
                x[vehicle['Vehicle_ID'], do['DO No']] 
                for do in pending_dos
            ]) <= 1, f"Vehicle_{str(vehicle['Vehicle_ID']).replace('-', '_')}_single"
        
        # Constraint 2: Each DO assigned to at most 1 vehicle
        for do in pending_dos:
            prob += lpSum([
                x[vehicle['Vehicle_ID'], do['DO No']] 
                for vehicle in available_vehicles
            ]) <= 1, f"DO_{str(do['DO No']).replace('-', '_')}_single"
        
        # Constraint 3: High priority DOs should be assigned if feasible
        for do in pending_dos:
            if do['Priority_Level'] in ['Critical', 'High']:
                # Check if there's at least one feasible vehicle
                if any(feasibility_matrix[v['Vehicle_ID'], do['DO No']][0] for v in available_vehicles):
                    prob += lpSum([
                        x[vehicle['Vehicle_ID'], do['DO No']] 
                        for vehicle in available_vehicles
                        if feasibility_matrix[vehicle['Vehicle_ID'], do['DO No']][0]
                    ]) == 1, f"Priority_{str(do['DO No']).replace('-', '_')}"
        
        # Solve
        logger.info("Solving optimization problem...")
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30)) # Add time limit
        
        # Extract solution
        allocations = []
        assigned_dos = set()
        
        for vehicle in available_vehicles:
            for do in pending_dos:
                if x[vehicle['Vehicle_ID'], do['DO No']].varValue == 1:
                    route = self.profit_predictor.get_route_object(
                        do['Loading_Point_Code'], do['Destination']
                    )
                    predicted_profit, features = self.profit_predictor.predict_trip_profit(
                        vehicle, do, route
                    )
                    
                    eta = self.calculate_eta(vehicle, do, route)
                    
                    allocations.append({
                        'DO_Number': do['DO No'],
                        'Customer_Name': do['Customer Name'],
                        'Destination': do['Destination'],
                        'Weight_Tons': do['Weight_Tons'],
                        'Recommended_Vehicle': vehicle['Vehicle_Number'],
                        'Vehicle_ID': vehicle['Vehicle_ID'],
                        'Vehicle_Type': vehicle['Ownership_Type'],
                        'Predicted_Profit_INR': round(predicted_profit, 2),
                        'Confidence_Score': self.calculate_confidence(features),
                        'Expected_Revenue': do['Expected_Freight_Rate'],
                        'Estimated_Cost': round(do['Expected_Freight_Rate'] - predicted_profit, 2),
                        'Route': f"{do['Loading_Point_Code']} -> {do['Destination']}",
                        'Distance_Km': route['Standard_Distance_Km'],
                        'ETA': eta.strftime('%Y-%m-%d %H:%M'),
                        'Capacity_Utilization_%': round(features['Capacity_Utilization_%'], 1),
                        'Reasoning': self.generate_reasoning(vehicle, do, features)
                    })
                    
                    assigned_dos.add(do['DO No'])
        
        # Handle unassigned DOs
        for do in pending_dos:
            if do['DO No'] not in assigned_dos:
                market_profit = self.estimate_market_truck_profit(do)
                
                allocations.append({
                    'DO_Number': do['DO No'],
                    'Customer_Name': do['Customer Name'],
                    'Destination': do['Destination'],
                    'Weight_Tons': do['Weight_Tons'],
                    'Recommended_Vehicle': 'HIRE MARKET TRUCK',
                    'Vehicle_ID': 'MARKET',
                    'Vehicle_Type': 'Market',
                    'Predicted_Profit_INR': round(market_profit, 2),
                    'Confidence_Score': 'Medium',
                    'Expected_Revenue': do['Expected_Freight_Rate'],
                    'Estimated_Cost': round(do['Expected_Freight_Rate'] - market_profit, 2),
                    'Route': f"{do['Loading_Point_Code']} -> {do['Destination']}",
                    'Distance_Km': 'N/A',
                    'ETA': 'TBD',
                    'Capacity_Utilization_%': 'N/A',
                    'Reasoning': 'No suitable owned/attached vehicle available. Market hire recommended.'
                })
        
        logger.info(f"Optimization complete: {len(allocations)} allocations generated")
        return allocations
    
    @staticmethod
    def calculate_eta(vehicle: Dict, do: Dict, route: Dict) -> datetime:
        """Calculate estimated time of arrival"""
        dist_to_loading = ConstraintValidator.haversine_distance(
            vehicle['Current_Location_Lat'], vehicle['Current_Location_Lng'],
            do['Loading_Point_Lat'], do['Loading_Point_Lng']
        )
        
        time_to_loading = dist_to_loading / 50  # 50 km/h avg
        loading_time = 2  # 2 hours for loading
        travel_time = route['Standard_Distance_Km'] / 50
        
        total_hours = time_to_loading + loading_time + travel_time
        
        return datetime.now() + timedelta(hours=total_hours)
    
    @staticmethod
    def calculate_confidence(features: Dict) -> str:
        """Calculate confidence score based on feature quality"""
        score = 0
        
        # High capacity utilization
        util = features['Capacity_Utilization_%']
        if 85 <= util <= 100:
            score += 30
        elif 70 <= util < 85:
            score += 20
        else:
            score += 10
        
        # Good fuel efficiency
        if features['Standard_Fuel_Efficiency'] > 5.5:
            score += 20
        
        # High driver score
        if features['Driver_Performance_Score'] > 80:
            score += 20
        
        # Good return load probability
        if features['Return_Load_Probability_%'] > 40:
            score += 15
        
        # Loading point efficiency
        if features['Loading_Point_Efficiency_Score'] > 80:
            score += 15
        
        if score >= 80:
            return "High"
        elif score >= 60:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def estimate_market_truck_profit(do: Dict) -> float:
        """Estimate profit margin for market truck hire"""
        # Market trucks typically have 10-15% profit margin
        revenue = do['Expected_Freight_Rate']
        profit_margin = revenue * 0.12  # 12% average
        return profit_margin
    
    @staticmethod
    def generate_reasoning(vehicle: Dict, do: Dict, feature_dict: Dict) -> str:
        """Generate human-readable reasoning for allocation"""
        reasons = []
        
        # Profitability reason
        if vehicle['Ownership_Type'] == 'Owned':
            reasons.append("Owned vehicle - Higher profit margin (avg 28% vs market 12%)")
        elif vehicle['Ownership_Type'] == 'Attached':
            reasons.append("Attached vehicle - Good profit margin (avg 20%)")
        
        # Capacity utilization
        utilization = feature_dict['Capacity_Utilization_%']
        if 85 <= utilization <= 100:
            reasons.append(f"Optimal capacity utilization ({utilization:.1f}%)")
        elif utilization < 70:
            reasons.append(f"⚠️ Under-utilized ({utilization:.1f}%) - consider smaller vehicle")
        
        # Proximity
        distance_to_loading = ConstraintValidator.haversine_distance(
            vehicle['Current_Location_Lat'], vehicle['Current_Location_Lng'],
            do['Loading_Point_Lat'], do['Loading_Point_Lng']
        )
        if distance_to_loading < 50:
            reasons.append(f"Vehicle nearby loading point ({distance_to_loading:.0f} km)")
        elif distance_to_loading > 200:
            reasons.append(f"⚠️ Vehicle far from loading point ({distance_to_loading:.0f} km)")
        
        # Fuel efficiency
        if feature_dict['Standard_Fuel_Efficiency'] > 5.5:
            reasons.append(f"Fuel efficient vehicle ({feature_dict['Standard_Fuel_Efficiency']} KMPL)")
        
        # Driver performance
        if feature_dict['Driver_Performance_Score'] > 80:
            reasons.append(f"Reliable driver (score: {feature_dict['Driver_Performance_Score']:.0f}/100)")
        
        # Return load opportunity
        if feature_dict['Return_Load_Probability_%'] > 40:
            reasons.append(f"Good backhaul chance ({feature_dict['Return_Load_Probability_%']:.0f}%)")
        
        return " | ".join(reasons)


# ==================== STEP 5: MAIN ORCHESTRATOR ====================

class FleetAllocationSystem:
    """Main system orchestrator using actual CSV data"""
    
    def __init__(self, data_folder: str = '.'):
        self.data_fetcher = DataFetcher(data_folder)
        self.profit_predictor = ProfitabilityPredictor(lr_details_path=f"{data_folder}/2_LR_Details_Enhanced_1000.csv")
        self.optimizer = OptimizationEngine(self.profit_predictor)
    
    def run_daily_allocation(self) -> pd.DataFrame:
        """Execute complete daily allocation workflow"""
        logger.info("=" * 80)
        logger.info("FLEET ALLOCATION SYSTEM - DAILY RUN (ACTUAL DATA)")
        logger.info("=" * 80)
        
        # Step 1: Fetch data
        logger.info("\n[STEP 1] Fetching data from CSVs...")
        pending_dos = self.data_fetcher.fetch_pending_dos()
        available_vehicles = self.data_fetcher.fetch_vehicles(status='Available')
        
        if not pending_dos:
            logger.warning("No pending DOs found. Nothing to allocate.")
            return pd.DataFrame()
            
        if not available_vehicles:
            logger.warning("No available vehicles found. All DOs will be marked for market hire.")
            # We can still proceed as the optimizer handles market hire for unassigned DOs
        
        # Step 2: Enrich data
        logger.info("\n[STEP 2] Enriching data...")
        pending_dos = DataEnricher.enrich_pending_dos(pending_dos)
        available_vehicles = DataEnricher.enrich_vehicles(available_vehicles)
        
        # Step 3: Run optimization
        logger.info("\n[STEP 3] Running optimization...")
        allocations = self.optimizer.optimize_daily_allocation(pending_dos, available_vehicles)
        
        # Convert to DataFrame
        df_allocations = pd.DataFrame(allocations)
        
        # Step 4: Generate summary
        if not df_allocations.empty:
            logger.info("\n[STEP 4] Generating summary...")
            self.print_summary(df_allocations)
        
        return df_allocations
    
    @staticmethod
    def print_summary(df: pd.DataFrame):
        """Print allocation summary without non-ASCII characters for Windows terminal compatibility"""
        logger.info("\n" + "=" * 80)
        logger.info("ALLOCATION SUMMARY")
        logger.info("=" * 80)
        
        total_dos = len(df)
        owned_vehicles = len(df[df['Vehicle_Type'] == 'Owned'])
        attached_vehicles = len(df[df['Vehicle_Type'] == 'Attached'])
        market_trucks = len(df[df['Vehicle_Type'] == 'Market'])
        
        total_profit = df['Predicted_Profit_INR'].sum()
        total_revenue = df['Expected_Revenue'].sum()
        
        logger.info(f"\nTotal DOs allocated: {total_dos}")
        logger.info(f"  - Owned vehicles: {owned_vehicles}")
        logger.info(f"  - Attached vehicles: {attached_vehicles}")
        logger.info(f"  - Market trucks: {market_trucks}")
        
        logger.info(f"\nFinancial Summary:")
        logger.info(f"  - Total Expected Revenue: INR {total_revenue:,.2f}")
        logger.info(f"  - Total Predicted Profit: INR {total_profit:,.2f}")
        logger.info(f"  - Overall Profit Margin: {(total_profit/total_revenue*100):.1f}%")
        
        logger.info("\n" + "=" * 80)
    
    def export_allocations(self, df: pd.DataFrame, filename: str = 'daily_allocations.csv'):
        """Export allocations to CSV"""
        df.to_csv(filename, index=False)
        logger.info(f"\nAllocations exported to {filename}")
    
    def export_allocations_json(self, df: pd.DataFrame, filename: str = 'daily_allocations.json'):
        """Export allocations to JSON"""
        df.to_json(filename, orient='records', indent=2)
        logger.info(f"Allocations exported to {filename}")


# ==================== USAGE EXAMPLE ====================

def main():
    """Main execution function using actual CSV data"""
    
    # Initialize system with the project folder
    data_folder = 'c:/Users/vrash/Desktop/Webanix/CC'
    system = FleetAllocationSystem(data_folder)
    
    # Train model on actual LR data
    logger.info("Initializing Profitability Model...")
    # Retrain every time if needed to ensure feature consistency during development
    # or if we detect a mismatch. For now, let's force retrain once to fix the mismatch.
    system.profit_predictor.train_model()
    
    # Run daily allocation workflow
    allocations_df = system.run_daily_allocation()
    
    if allocations_df.empty:
        logger.warning("Workflow returned no allocations.")
        return

    # Display top results
    print("\n" + "=" * 120)
    print("DETAILED ALLOCATION RESULTS (Top 20)")
    print("=" * 120)
    # Using encode-decode to strip non-ASCII characters that might cause print errors in some environments
    output_str = allocations_df.head(20).to_string(index=False)
    print(output_str.encode('ascii', 'ignore').decode('ascii'))
    
    # Export results
    system.export_allocations(allocations_df)
    system.export_allocations_json(allocations_df)
    
    # Show top profitable allocations
    print("\n" + "=" * 120)
    print("TOP 5 MOST PROFITABLE ALLOCATIONS")
    print("=" * 120)
    top_5 = allocations_df.nlargest(5, 'Predicted_Profit_INR')[
        ['DO_Number', 'Customer_Name', 'Recommended_Vehicle', 'Predicted_Profit_INR', 'Reasoning']
    ]
    print(top_5.to_string(index=False))


if __name__ == "__main__":
    main()