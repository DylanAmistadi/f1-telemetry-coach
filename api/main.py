from telemetry_engine.processing import align_telemetry_and_calculate_delta
from telemetry_engine.coaching_logic import identify_corners, analyze_corner, generate_coaching_feedback, analyze_straight, generate_straight_feedback

def run_telemetry_pipeline(lap_reference, lap_compare):
    """
    Executes the math and coaching logic pipeline on two FastF1 Lap objects.
    
    Args:
        lap_reference (fastf1.core.Lap): The reference lap (faster driver).
        lap_compare (fastf1.core.Lap): The comparison lap.
        
    Returns:
        tuple: (aligned_df, feedbacks_list)
    """
    # 1. Spatial/Micro-sector alignment
    aligned_df = align_telemetry_and_calculate_delta(lap_reference, lap_compare, num_sectors=25)
    
    # 2. Identify Action Zones (Corners)
    corners = identify_corners(aligned_df)
    
    feedbacks = []
    
    # 3. Analyze each Action Zone and the Straight following it
    corner_count = 1
    for i, corner_df in enumerate(corners):
        # Determine the start of the next zone to compute cause-and-effect time loss down the straight
        next_dist = corners[i+1]['Distance'].iloc[0] if i + 1 < len(corners) else None
        
        # Analyze Corner
        metrics = analyze_corner(corner_df, next_zone_start_dist=next_dist, full_df=aligned_df)
        feedback_text = generate_coaching_feedback(metrics)
        
        feedbacks.append({
            'start_dist': metrics['start_distance'],
            'end_dist': metrics['end_distance'],
            'feedback': feedback_text,
            'type': 'corner',
            'index': corner_count
        })
        corner_count += 1
        
        # Analyze Straight
        if next_dist is not None:
            end_dist = corner_df['Distance'].iloc[-1]
            # Straight is the segment between end of this corner and start of next
            straight_mask = (aligned_df['Distance'] > end_dist) & (aligned_df['Distance'] < next_dist)
            df_straight = aligned_df[straight_mask]
            
            if not df_straight.empty:
                straight_metrics = analyze_straight(df_straight)
                straight_text = generate_straight_feedback(straight_metrics)
                
                if straight_text:
                    feedbacks.append({
                        'start_dist': straight_metrics['start_dist'],
                        'end_dist': straight_metrics['end_dist'],
                        'feedback': straight_text,
                        'type': 'straight',
                        'index': None
                    })
        
    return aligned_df, feedbacks
