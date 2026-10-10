import numpy as np
import pandas as pd

def align_telemetry_and_calculate_delta(lap_reference, lap_compare, num_sectors=25):
    """
    Aligns the telemetry data of two different laps based on distance, utilizing micro-sectors 
    matched by X/Y coordinates to prevent distortion from driving different lines (the 'wider line' issue).
    
    Args:
        lap_reference (fastf1.core.Lap): The reference lap (usually the faster lap).
        lap_compare (fastf1.core.Lap): The lap to compare against the reference.
        num_sectors (int): Number of micro-sectors to split the lap into for alignment.
        
    Returns:
        pd.DataFrame: A DataFrame containing aligned telemetry and Delta Time.
    """
    tel_ref = lap_reference.get_telemetry()
    tel_comp = lap_compare.get_telemetry()
    
    # 1. Micro-sector definition based on reference lap indices
    ref_boundaries_idx = np.linspace(0, len(tel_ref) - 1, num_sectors + 1).astype(int)
    
    # 2. Match micro-sectors on the comparison lap using spatial coordinates (X, Y)
    comp_boundaries_idx = [0]
    
    x_comp = tel_comp['X'].values
    y_comp = tel_comp['Y'].values
    
    for idx in ref_boundaries_idx[1:-1]:
        x_ref_target = tel_ref['X'].iloc[idx]
        y_ref_target = tel_ref['Y'].iloc[idx]
        
        # Euclidean distance to find the closest physical point on the track
        distances = (x_comp - x_ref_target)**2 + (y_comp - y_ref_target)**2
        closest_idx = np.argmin(distances)
        comp_boundaries_idx.append(closest_idx)
        
    comp_boundaries_idx.append(len(tel_comp) - 1)
    
    # Ensure strict monotonicity for slicing
    for i in range(1, len(comp_boundaries_idx)):
        if comp_boundaries_idx[i] <= comp_boundaries_idx[i-1]:
            comp_boundaries_idx[i] = comp_boundaries_idx[i-1] + 1
            if comp_boundaries_idx[i] >= len(tel_comp):
                comp_boundaries_idx[i] = len(tel_comp) - 1
                
    # Detect steering column if available
    steer_col = 'SteeringAngle' if 'SteeringAngle' in tel_ref.columns else 'Steering' if 'Steering' in tel_ref.columns else None

    aligned_chunks = []
    
    # 3. Process each micro-sector independently
    for i in range(num_sectors):
        ref_start, ref_end = ref_boundaries_idx[i], ref_boundaries_idx[i+1]
        comp_start, comp_end = comp_boundaries_idx[i], comp_boundaries_idx[i+1]
        
        chunk_ref = tel_ref.iloc[ref_start:ref_end+1]
        chunk_comp = tel_comp.iloc[comp_start:comp_end+1]
        
        dist_ref = chunk_ref['Distance'].values
        dist_comp = chunk_comp['Distance'].values
        
        # Local normalization within this specific micro-sector (0.0 to 1.0)
        norm_ref = (dist_ref - dist_ref[0]) / (dist_ref[-1] - dist_ref[0] + 1e-6)
        norm_comp = (dist_comp - dist_comp[0]) / (dist_comp[-1] - dist_comp[0] + 1e-6)
        
        num_points = len(chunk_ref)
        common_norm = np.linspace(0, 1.0, num_points)
        
        # Scale back to reference distance mapping
        final_dist = dist_ref[0] + common_norm * (dist_ref[-1] - dist_ref[0])
        
        # Interpolate core signals
        speed_ref = np.interp(common_norm, norm_ref, chunk_ref['Speed'].values)
        throttle_ref = np.interp(common_norm, norm_ref, chunk_ref['Throttle'].values)
        brake_ref = np.interp(common_norm, norm_ref, chunk_ref['Brake'].astype(float).values)
        time_ref = np.interp(common_norm, norm_ref, chunk_ref['Time'].dt.total_seconds().values)
        gear_ref = np.interp(common_norm, norm_ref, chunk_ref['nGear'].values)
        
        # Track map coordinates from reference
        x_ref = np.interp(common_norm, norm_ref, chunk_ref['X'].values)
        y_ref = np.interp(common_norm, norm_ref, chunk_ref['Y'].values)
        
        speed_comp = np.interp(common_norm, norm_comp, chunk_comp['Speed'].values)
        throttle_comp = np.interp(common_norm, norm_comp, chunk_comp['Throttle'].values)
        brake_comp = np.interp(common_norm, norm_comp, chunk_comp['Brake'].astype(float).values)
        time_comp = np.interp(common_norm, norm_comp, chunk_comp['Time'].dt.total_seconds().values)
        gear_comp = np.interp(common_norm, norm_comp, chunk_comp['nGear'].values)
        
        # Steering interpolation (safe handling if missing)
        if steer_col:
            steer_ref = np.interp(common_norm, norm_ref, chunk_ref[steer_col].values)
            steer_comp = np.interp(common_norm, norm_comp, chunk_comp[steer_col].values)
        else:
            steer_ref = np.zeros(num_points)
            steer_comp = np.zeros(num_points)
            
        delta_time = time_comp - time_ref
        
        # Calculate who gained time in this micro-sector
        sector_time_diff = delta_time[-1] - delta_time[0]
        # -1 for Reference dominance, 1 for Comparison dominance
        dominance = 1.0 if sector_time_diff < 0 else -1.0
        
        df_chunk = pd.DataFrame({
            'Distance': final_dist,
            'X': x_ref,
            'Y': y_ref,
            'Speed_Ref': speed_ref,
            'Speed_Comp': speed_comp,
            'Brake_Ref': brake_ref,
            'Brake_Comp': brake_comp,
            'Throttle_Ref': throttle_ref,
            'Throttle_Comp': throttle_comp,
            'Gear_Ref': np.round(gear_ref).astype(int),
            'Gear_Comp': np.round(gear_comp).astype(int),
            'Steer_Ref': steer_ref,
            'Steer_Comp': steer_comp,
            'Delta_Time': delta_time,
            'MicroSector_Dominance': dominance
        })
        
        # Drop the last row to avoid duplicates when concatenating adjacent chunks
        if i < num_sectors - 1:
            df_chunk = df_chunk.iloc[:-1]
            
        aligned_chunks.append(df_chunk)
        
    # 4. Stitch back together
    aligned_data = pd.concat(aligned_chunks, ignore_index=True)
    aligned_data['Speed_Diff'] = aligned_data['Speed_Comp'] - aligned_data['Speed_Ref']
    return aligned_data
