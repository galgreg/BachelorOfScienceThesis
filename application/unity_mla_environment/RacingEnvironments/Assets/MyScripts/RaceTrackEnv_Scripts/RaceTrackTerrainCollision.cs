using UnityEngine;

public class RaceTrackTerrainCollision : MonoBehaviour {
    // private const float PENALTY_FOR_COLLISION = -10.0f;
    private const float PENALTY_FOR_COLLISION = 0.0f;
    
    void OnCollisionEnter(Collision aCarCollision) {
        var carObject = aCarCollision.gameObject;
        var carAgentComponent = carObject.GetComponent<CarAgent>();
        if (carAgentComponent != null) {
            carAgentComponent.AddReward(PENALTY_FOR_COLLISION);
            carAgentComponent.SaveEpisodeReward();
            carAgentComponent.Done();
        }
    }
}
