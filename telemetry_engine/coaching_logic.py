import pandas as pd
import numpy as np

def detect_snap_oversteer(steering_trace):
    """
    Detects if a driver suffered a snap oversteer / traction loss 
    by analyzing high-frequency reversals in their independent steering trace.
    """
    # 1. Smooth trace to ignore microscopic noise/sensor jitter
    smoothed = steering_trace.rolling(window=3, min_periods=1, center=True).mean()
    
    # 2. First derivative (steering velocity)
    steer_vel = smoothed.diff().fillna(0)
    
    # 3. Filter out slow/tiny movements (deadzone)
    steer_vel_filtered = steer_vel.copy()
    steer_vel_filtered[steer_vel_filtered.abs() < 2.0] = 0
    
    # 4. Count sign changes (reversals in wheel direction)
    non_zero_vel = steer_vel_filtered[steer_vel_filtered != 0]
    if len(non_zero_vel) == 0:
        return False
        
    signs = np.sign(non_zero_vel.values)
    reversals = np.sum(np.diff(signs) != 0)
    
    # A single corner has 1 reversal. A chicane has 2. 
    # 3 or more strictly indicates sawing at the wheel (correction).
    if reversals >= 3:
        return True
        
    # 5. Check for extreme steering velocity (Panic correction)
    # Even if they only corrected once and didn't saw, the speed of the wheel is immense.
    max_vel = steer_vel.abs().max()
    if max_vel > 25.0:
        return True
        
    return False

def identify_corners(df, sustained_distance=20.0):
    """
    Identifies 'Action Zones' (corners or lift-off zones) based on pedal inputs.
    """
    off_throttle = (df['Throttle_Ref'] < 99) | (df['Throttle_Comp'] < 99) | \
                   (df['Brake_Ref'] > 0) | (df['Brake_Comp'] > 0)
                   
    full_throttle = (df['Throttle_Ref'] >= 99) & (df['Throttle_Comp'] >= 99)
    
    action_zones = []
    in_zone = False
    start_idx = 0
    full_throttle_start_dist = None
    distance = df['Distance'].values
    
    for i in range(len(df)):
        if not in_zone:
            if off_throttle.iloc[i]:
                in_zone = True
                start_idx = i
                full_throttle_start_dist = None
        else:
            if full_throttle.iloc[i]:
                if full_throttle_start_dist is None:
                    full_throttle_start_dist = distance[i]
                elif (distance[i] - full_throttle_start_dist) >= sustained_distance:
                    end_idx = i
                    action_zones.append(df.iloc[start_idx:end_idx].copy())
                    in_zone = False
                    full_throttle_start_dist = None
            else:
                full_throttle_start_dist = None
                
    if in_zone:
        action_zones.append(df.iloc[start_idx:].copy())
        
    return action_zones


def analyze_corner(df_corner, next_zone_start_dist=None, full_df=None):
    """
    Extracts key performance metrics from a single corner (Action Zone).
    """
    # 1. V-Min Delta
    vmin_ref = df_corner['Speed_Ref'].min()
    vmin_comp = df_corner['Speed_Comp'].min()
    vmin_delta = vmin_comp - vmin_ref
    
    # 2. V-Min Position Delta
    dist_vmin_ref = df_corner.loc[df_corner['Speed_Ref'].idxmin(), 'Distance']
    dist_vmin_comp = df_corner.loc[df_corner['Speed_Comp'].idxmin(), 'Distance']
    vmin_pos_delta = dist_vmin_comp - dist_vmin_ref
    
    # Minimum Throttle Position Delta
    dist_min_thr_ref = df_corner.loc[df_corner['Throttle_Ref'].idxmin(), 'Distance']
    dist_min_thr_comp = df_corner.loc[df_corner['Throttle_Comp'].idxmin(), 'Distance']
    min_thr_pos_delta = dist_min_thr_comp - dist_min_thr_ref

    # 3. Minimum Throttle & Max Brake
    min_thr_ref = df_corner['Throttle_Ref'].min()
    min_thr_comp = df_corner['Throttle_Comp'].min()
    
    max_brake_ref = df_corner['Brake_Ref'].max()
    max_brake_comp = df_corner['Brake_Comp'].max()
    
    # 4. Time gained/lost calculation over the ensuing straight
    start_delta = df_corner['Delta_Time'].iloc[0]
    if next_zone_start_dist is not None and full_df is not None:
        next_idx = (full_df['Distance'] == next_zone_start_dist).idxmax()
        end_delta = full_df['Delta_Time'].loc[next_idx]
    elif full_df is not None:
        end_delta = full_df['Delta_Time'].iloc[-1]
    else:
        end_delta = df_corner['Delta_Time'].iloc[-1]
    time_lost = end_delta - start_delta
    
    # 5. Throttle Hesitation Check (Snap oversteer via pedals)
    throttle_diff_comp = df_corner['Throttle_Comp'].diff()
    throttle_diff_ref = df_corner['Throttle_Ref'].diff()
    comp_had_hesitation = ((throttle_diff_comp < -10) & (df_corner['Brake_Comp'] == 0)).any()
    ref_had_hesitation = ((throttle_diff_ref < -10) & (df_corner['Brake_Ref'] == 0)).any()
    
    # 6. Minimum Gear
    min_gear_ref = df_corner['Gear_Ref'].min()
    min_gear_comp = df_corner['Gear_Comp'].min()
    
    # 7. Trail Braking & Initial Bite
    brake_diff_ref = df_corner['Brake_Ref'].diff().fillna(0)
    brake_diff_comp = df_corner['Brake_Comp'].diff().fillna(0)
    
    initial_bite_ref = brake_diff_ref.max()
    initial_bite_comp = brake_diff_comp.max()
    
    trail_mask_ref = (brake_diff_ref < 0) & (df_corner['Brake_Ref'] > 0)
    trail_mask_comp = (brake_diff_comp < 0) & (df_corner['Brake_Comp'] > 0)
    
    trail_slope_ref = brake_diff_ref[trail_mask_ref].mean() if trail_mask_ref.any() else 0
    trail_slope_comp = brake_diff_comp[trail_mask_comp].mean() if trail_mask_comp.any() else 0
    
    # 8. Steering Correlation (Power while steering)
    max_steer_corner = df_corner['Steer_Ref'].abs().max()
    early_power_high_steer = False
    
    ref_snap = False
    comp_snap = False
    
    if max_steer_corner > 0:
        # Snap Detection via Steering
        ref_snap = detect_snap_oversteer(df_corner['Steer_Ref'])
        comp_snap = detect_snap_oversteer(df_corner['Steer_Comp'])
        
        # Early Power Detection
        comp_power_idx = df_corner['Throttle_Comp'] > 50
        if comp_power_idx.any():
            steer_at_power_comp = df_corner.loc[comp_power_idx, 'Steer_Comp'].abs()
            steer_at_power_ref = df_corner.loc[df_corner['Throttle_Ref'] > 50, 'Steer_Ref'].abs()
            if not steer_at_power_comp.empty and not steer_at_power_ref.empty:
                max_steer_power_comp = steer_at_power_comp.max()
                max_steer_power_ref = steer_at_power_ref.max()
                if max_steer_power_comp > (max_steer_power_ref + 15):
                    early_power_high_steer = True
                    
    # 9. Braking Point Start Delta
    brake_points_ref = df_corner.loc[df_corner['Brake_Ref'] > 0, 'Distance']
    brake_points_comp = df_corner.loc[df_corner['Brake_Comp'] > 0, 'Distance']
    
    dist_brake_start_ref = brake_points_ref.min() if not brake_points_ref.empty else np.nan
    dist_brake_start_comp = brake_points_comp.min() if not brake_points_comp.empty else np.nan
    
    if pd.isna(dist_brake_start_ref) or pd.isna(dist_brake_start_comp):
        brake_start_diff = 0.0
    else:
        # Positive means Comp braked later (further down the track)
        brake_start_diff = dist_brake_start_comp - dist_brake_start_ref
    
    metrics = {
        'start_distance': df_corner['Distance'].iloc[0],
        'end_distance': df_corner['Distance'].iloc[-1],
        'vmin_delta': vmin_delta,
        'vmin_pos_delta': vmin_pos_delta,
        'min_thr_ref': min_thr_ref,
        'min_thr_comp': min_thr_comp,
        'min_thr_pos_delta': min_thr_pos_delta,
        'max_brake_ref': max_brake_ref,
        'max_brake_comp': max_brake_comp,
        'time_lost': time_lost,
        'comp_had_hesitation': bool(comp_had_hesitation),
        'ref_had_hesitation': bool(ref_had_hesitation),
        'min_gear_ref': min_gear_ref,
        'min_gear_comp': min_gear_comp,
        'initial_bite_ref': initial_bite_ref,
        'initial_bite_comp': initial_bite_comp,
        'trail_slope_ref': trail_slope_ref,
        'trail_slope_comp': trail_slope_comp,
        'early_power_high_steer': early_power_high_steer,
        'ref_snap': ref_snap,
        'comp_snap': comp_snap,
        'brake_start_diff': brake_start_diff
    }
    
    return metrics


def generate_coaching_feedback(metrics):
    """
    Uses rule-based logic to output quantitative and prescriptive coaching feedback strings based on corner metrics.
    """
    feedback = []
    
    comp_snap = metrics.get('comp_snap', False)
    ref_snap = metrics.get('ref_snap', False)
    t_lost = metrics['time_lost']
    
    # Rule G: Dual Perspective Steering Snap Detection
    if comp_snap and not ref_snap:
        feedback.append(f"Steering Snap: You applied sudden opposite lock mid-corner (loss of rear grip). Cost: {max(0, t_lost):.3f}s. Action: Ensure car is balanced before exit phase.")
    elif not comp_snap and ref_snap:
        feedback.append(f"Reference Error Capitalized: You drove smoothly while the reference suffered a snap oversteer. Gained: {max(0, -t_lost):.3f}s.")
    elif comp_snap and ref_snap:
        feedback.append("Both drivers struggled with rear grip and applied steering corrections on exit.")
    
    # Time Loss specific rules
    if t_lost > 0:
        
        # Rule D: Gear Selection
        if metrics['min_gear_comp'] != metrics['min_gear_ref']:
            feedback.append(
                f"Gear Selection: You took this corner in gear {metrics['min_gear_comp']} while Reference used gear {metrics['min_gear_ref']}. "
                f"Cost: {t_lost:.3f}s. Action: Adjust gear selection to match reference for optimal engine braking and exit torque."
            )
            
        # Rule E: Braking Phase
        if metrics['max_brake_comp'] > 20 and metrics['max_brake_ref'] > 20:
            if metrics['initial_bite_comp'] < (metrics['initial_bite_ref'] * 0.7):
                feedback.append(
                    f"Braking Technique: Soft initial brake bite ({metrics['initial_bite_comp']:.1f}%/m vs Ref's {metrics['initial_bite_ref']:.1f}%/m). "
                    f"Cost: {t_lost:.3f}s. Action: Hit the brake pedal harder initially to maximize deceleration."
                )
            elif metrics['trail_slope_comp'] < (metrics['trail_slope_ref'] * 1.5):
                feedback.append(
                    f"Braking Technique: Your trail brake release was abruptly steep at {abs(metrics['trail_slope_comp']):.1f}%/m compared to the reference's {abs(metrics['trail_slope_ref']):.1f}%/m. "
                    f"Cost: {t_lost:.3f}s. Action: Smoother brake release upon turn-in to avoid upsetting the rear."
                )
                
        # Rule F: Steering Correlation (Applying power early)
        if metrics['early_power_high_steer']:
            feedback.append(
                f"Early Power: Applying power too early before unwinding the wheel. "
                f"Cost: {t_lost:.3f}s. Action: Unwind the steering wheel before aggressive throttle application."
            )

        # Rule A: Overshooting
        if metrics['vmin_pos_delta'] > 5.0 and metrics['min_thr_pos_delta'] > 0:
            brake_str = f"{metrics['brake_start_diff']:.1f}m later" if metrics['brake_start_diff'] > 0 else f"{abs(metrics['brake_start_diff']):.1f}m earlier"
            feedback.append(
                f"Overshooting: You braked {brake_str}, shifting your V-Min {metrics['vmin_pos_delta']:.1f}m forward and dropping {abs(metrics['vmin_delta']):.1f} km/h more than the reference. "
                f"Cost: {t_lost:.3f}s. Action: Brake earlier, hit the apex, and prioritize exit traction."
            )
            
        # Rule B: Lift-off corners
        if metrics['max_brake_ref'] <= 0 and metrics['max_brake_comp'] <= 0:
            if metrics['min_thr_comp'] < metrics['min_thr_ref']:
                feedback.append(
                    f"Over-slowing: You dropped the throttle to {metrics['min_thr_comp']:.1f}% while the reference maintained {metrics['min_thr_ref']:.1f}%. "
                    f"Cost: {t_lost:.3f}s. Action: Trust the aero and carry more minimum rolling speed."
                )
                
        # Rule C: Traction Instability / Throttle Hesitation
        if metrics['comp_had_hesitation'] and not metrics['ref_had_hesitation']:
            feedback.append(
                f"Traction Instability: You had a sudden throttle lift on exit (snap oversteer correction). "
                f"Cost: {t_lost:.3f}s. Action: Wait for the car to rotate fully before aggressively applying throttle."
            )
                
        # Fallback if no specific rule matched but time was lost
        if not feedback:
            feedback.append(f"General deficit: You lost {t_lost:.3f}s in this sector, but it didn't trigger a specific driving error pattern.")
            
    else:
        # Time Gained (or Neutral) specific logic
        if not feedback:
            feedback.append(f"Good execution: You gained {abs(t_lost):.3f}s in this sector compared to the reference driver.")
            
    return " ".join(feedback)

def analyze_straight(df_straight):
    """
    Extracts metrics for a straight-line segment (between two corners).
    """
    if len(df_straight) < 10:
        return None
        
    start_speed_ref = df_straight['Speed_Ref'].iloc[0]
    start_speed_comp = df_straight['Speed_Comp'].iloc[0]
    start_speed_diff = start_speed_comp - start_speed_ref
    
    end_speed_ref = df_straight['Speed_Ref'].iloc[-1]
    end_speed_comp = df_straight['Speed_Comp'].iloc[-1]
    end_speed_diff = end_speed_comp - end_speed_ref
    
    # Positive swing = Comp gained relative speed over the length of the straight
    speed_swing = end_speed_diff - start_speed_diff
    
    time_lost_straight = df_straight['Delta_Time'].iloc[-1] - df_straight['Delta_Time'].iloc[0]
    
    metrics = {
        'start_dist': df_straight['Distance'].iloc[0],
        'end_dist': df_straight['Distance'].iloc[-1],
        'start_speed_diff': start_speed_diff,
        'end_speed_diff': end_speed_diff,
        'speed_swing': speed_swing,
        'time_lost': time_lost_straight
    }
    return metrics

def generate_straight_feedback(metrics):
    """
    Generates coaching feedback for aerodynamic, slipstream, or engine clipping scenarios on straights.
    """
    if metrics is None:
        return None
        
    feedback = []
    
    # Scenario 1 (Tow rescuing a bad exit)
    # Comp exits slower (<-2 km/h) but gains massive top end (+4 km/h delta swing)
    if metrics['start_speed_diff'] <= -2.0 and metrics['speed_swing'] >= 4.0:
        feedback.append("Aerodynamic advantage detected. You had a deficit in corner exit traction, but overcame it with superior top-end acceleration (likely due to a slipstream or lower drag setup).")
        
    # Scenario 2 (Tow amplifying a good exit)
    elif metrics['start_speed_diff'] >= 1.0 and metrics['speed_swing'] >= 4.0:
        feedback.append("Compounded aerodynamic advantage. You carried superior exit speed out of the corner and the delta exponentially increased down the straight, indicating a slipstream or powerful aero setup on top of a good exit.")
        
    # Scenario 3 (Clipping/Drag issue)
    # Comp exits faster but bleeds the advantage away completely on the straight
    elif metrics['start_speed_diff'] >= 2.0 and metrics['speed_swing'] <= -4.0:
        feedback.append("High drag or ERS clipping detected. You had a superior corner exit, but the speed advantage vanished at the top end of the straight.")

    if feedback:
        return " ".join(feedback)
    return None
