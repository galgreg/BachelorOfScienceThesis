using UnityEngine;

public class RaceTrackTerrainCollision : MonoBehaviour {
    void OnCollisionEnter(Collision aCarCollision) {
        var carObject = aCarCollision.gameObject;
        var carAgentComponent = carObject.GetComponent<CarAgent>();
        if (carAgentComponent != null) {
            carAgentComponent.SaveEpisodeReward();
            carAgentComponent.Done();
        }
    }
}
