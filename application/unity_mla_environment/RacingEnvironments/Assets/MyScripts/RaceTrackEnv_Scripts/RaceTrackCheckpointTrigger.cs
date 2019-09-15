using System.Collections.Generic;
using UnityEngine;

public class RaceTrackCheckpointTrigger : MonoBehaviour {
    private const float REWARD_FOR_CHECKPOINT = 1.0f;
    private HashSet<int> mAgentKeySet;

    void Start() {
        mAgentKeySet = new HashSet<int>();
    }
    
    public void ResetKeySet() {
        mAgentKeySet.Clear();
    }

    void OnTriggerEnter(Collider aCarCollider) {
        var carObject = aCarCollider.gameObject;
        var carAgentComponent = carObject.GetComponent<CarAgent>();
        if (carAgentComponent != null) {
            var agentID = carObject.GetInstanceID();
            if (mAgentKeySet.Contains(agentID)) {
                carAgentComponent.AddReward(-REWARD_FOR_CHECKPOINT);
            } else {
                mAgentKeySet.Add(agentID);
                carAgentComponent.AddReward(REWARD_FOR_CHECKPOINT);
            }
        }
    }
}
