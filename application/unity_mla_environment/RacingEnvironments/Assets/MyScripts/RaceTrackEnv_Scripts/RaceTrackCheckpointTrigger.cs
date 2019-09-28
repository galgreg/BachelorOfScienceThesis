using System.Collections.Generic;
using UnityEngine;

public class RaceTrackCheckpointTrigger : MonoBehaviour {
    private const float REWARD_FOR_CHECKPOINT = 1.0f;

    void OnTriggerEnter(Collider aCarCollider) {
        var carObject = aCarCollider.gameObject;
        var carAgentComponent = carObject.GetComponent<CarAgent>();
        if (carAgentComponent != null) {
            carAgentComponent.SetReward(REWARD_FOR_CHECKPOINT);
        }
    }
}
