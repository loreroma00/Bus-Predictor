import csv
import logging
import re
from typing import Dict, List, Union
from .static_data import VehicleType, Engine, FuelType, EuroType

logger = logging.getLogger(__name__)

FUEL_MAPPING = {
    "Diesel": FuelType.DIESEL,
    "Electric": FuelType.ELECTRIC,
    "CNG (Compressed Natural Gas)": FuelType.CNG,
    "Hybrid [Diesel & Electricity]": FuelType.HYBRID_DIESEL_ELECTRIC,
    "Hybrid (Diesel & Electricity)": FuelType.HYBRID_DIESEL_ELECTRIC,
    "Electric [NMC Batteries]": FuelType.ELECTRIC_NMC,
    "Electric (NMC Battery)": FuelType.ELECTRIC_NMC,
    "Electrical [Lithium-Iron-Phosphate Batteries]": FuelType.ELECTRIC_LFP,
    "Electrical [CATL LFP Batteries]": FuelType.ELECTRIC_CATL_LFP,
    "Electric (Ni-NaCL Battery)": FuelType.ELECTRIC_NINACL,
    "Hybrid (Diesel & Electricity LTO Batteries)": FuelType.HYBRID_LTO,
    "Dual [Diesel & Electric]": FuelType.DUAL_DIESEL_ELECTRIC,
    "": FuelType.DIESEL # Default
}

def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val)) if val else default
    except (ValueError, TypeError):
        return default

def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default

def parse_id_ranges(range_str: str) -> List[str]:
    """
    Parses a string like "2001-2050;A300-A350;753_01-753_14" into a list of individual ID strings.
    Handles numeric ranges and alphanumeric ranges with consistent prefixes.
    Converts underscores to hyphens in the resulting IDs.
    """
    ids = []
    if not range_str:
        return ids
    
    parts = range_str.split(';')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                # Split start and end of range
                # We use rsplit to handle cases where prefix might contain hyphens (unlikely but safe)
                # Actually, standard split is fine as long as we know the format.
                start_s, end_s = part.split('-')
                
                # Normalize underscores to hyphens in the input parts for processing
                # but we need to keep track of the numeric part.
                
                # Extract prefix and number for start and end
                # Regex matches anything followed by a sequence of digits at the end
                match_start = re.match(r"(.*?)(\d+)$", start_s)
                match_end = re.match(r"(.*?)(\d+)$", end_s)
                
                if match_start and match_end and match_start.group(1) == match_end.group(1):
                    prefix = match_start.group(1).replace('_', '-')
                    s_num_str = match_start.group(2)
                    e_num_str = match_end.group(2)
                    
                    s_num = int(s_num_str)
                    e_num = int(e_num_str)
                    
                    # Determine padding width from the start string
                    width = len(s_num_str)
                    
                    for i in range(s_num, e_num + 1):
                        # Format with leading zeros if necessary
                        num_str = str(i).zfill(width)
                        ids.append(f"{prefix}{num_str}")
                else:
                    # Fallback for purely numeric if regex failed somehow (unlikely)
                    if start_s.isdigit() and end_s.isdigit():
                        ids.extend([str(i) for i in range(int(start_s), int(end_s) + 1)])
                    else:
                        logger.warning(f"Could not parse range: {part}")
            except Exception as e:
                logger.warning(f"Error parsing ID range format '{part}': {e}")
        else:
            # Single ID - still normalize underscore to hyphen
            ids.append(part.replace('_', '-'))
    return ids

def get_enum_value(enum_class, value_str):
    if not value_str: return None
    
    # Check manual mapping first if it exists
    if enum_class == FuelType:
        if value_str in FUEL_MAPPING:
            return FUEL_MAPPING[value_str]

    try:
        # Try exact match (integer or name)
        return enum_class(int(value_str)) if value_str.isdigit() else enum_class[value_str]
    except (ValueError, KeyError):
        # Try case-insensitive name match
        for name, member in enum_class.__members__.items():
            if name.upper() == value_str.upper().replace(" ", "_"):
                return member
        return None

def sanitize_str(val: str) -> str:
    """Translates underscores to hyphens."""
    if val:
        return val.replace('_', '-')
    return val

def parse_list(val: str) -> List[str]:
    """Parses a string separated by ; or , into a list of strings."""
    if not val: return []
    return [sanitize_str(x.strip()) for x in val.replace(';', ',').split(',') if x.strip()]

def load_fleet(csv_path: str) -> Dict[str, VehicleType]:
    """
    Loads fleet data from CSV and returns a dictionary mapping
    Bus ID (as string) to VehicleType.
    """
    fleet_map: Dict[str, VehicleType] = {}
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    # Parse Engine
                    fuel_str = row.get("fuel_type", "DIESEL")
                    fuel = get_enum_value(FuelType, fuel_str) or FuelType.DIESEL
                    
                    euro_str = row.get("euro_type", "EURO_6")
                    # Handle empty euro type
                    if not euro_str:
                        euro = EuroType.EURO_6
                    else:
                        euro = get_enum_value(EuroType, euro_str) or EuroType.EURO_6
                    
                    engine = Engine(
                        name=sanitize_str(row.get("engine_name", "Unknown")),
                        fuel=fuel,
                        euro=euro
                    )
                    
                    # Parse ID Ranges (now returns flattened list of strings)
                    id_list = parse_id_ranges(row.get("id_ranges", ""))
                    
                    # Parse and sanitize lists
                    deposits = parse_list(row.get("deposits", ""))
                    constructors = parse_list(row.get("constructors", ""))

                    # Create VehicleType
                    v_type = VehicleType(
                        name=sanitize_str(row.get("name")),
                        ids=[], # We don't store the complex range object anymore, or we could store the list
                        amount=safe_int(row.get("amount")),
                        active=safe_int(row.get("active")),
                        agency=sanitize_str(row.get("agency", "ATAC")),
                        deposits=deposits,
                        doors=safe_int(row.get("doors")),
                        engine=engine,
                        length=safe_float(row.get("length")),
                        width=safe_float(row.get("width")),
                        height=safe_float(row.get("height")),
                        weight=safe_float(row.get("weight")),
                        capacity_sitting=safe_int(row.get("capacity_sitting")),
                        capacity_standing=safe_int(row.get("capacity_standing")),
                        capacity_total=safe_int(row.get("capacity_total")),
                        construction_year=safe_int(row.get("construction_year")),
                        constructors=constructors
                    )
                    
                    # Populate map
                    for bus_id in id_list:
                        fleet_map[str(bus_id)] = v_type
                            
                except Exception as e:
                    logger.error(f"Error parsing vehicle row {row.get('name', 'Unknown')}: {e}")
                    continue
                    
    except FileNotFoundError:
        logger.error(f"Fleet CSV not found at {csv_path}")
    except Exception as e:
        logger.error(f"Error loading fleet: {e}")
        
    logger.info(f"Loaded fleet with {len(fleet_map)} mapped vehicles.")
    return fleet_map
