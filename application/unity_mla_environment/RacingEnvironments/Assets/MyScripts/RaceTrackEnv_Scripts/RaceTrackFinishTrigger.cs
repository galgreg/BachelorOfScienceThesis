using UnityEngine;

public class RaceTrackFinishTrigger : MonoBehaviour {
    private const float REWARD_FOR_FINISH = 10.0f;
    
    void OnTriggerEnter(Collider aCarCollider) {
        var carObject = aCarCollider.gameObject;
        var carAgentComponent = carObject.GetComponent<CarAgent>();
        if (carAgentComponent != null) {
            carAgentComponent.AddReward(REWARD_FOR_FINISH);
            carAgentComponent.SaveEpisodeReward();
            carAgentComponent.Done();
        }
    }
}
