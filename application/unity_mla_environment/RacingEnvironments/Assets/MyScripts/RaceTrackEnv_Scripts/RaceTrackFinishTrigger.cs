﻿using UnityEngine;
using MLAgents;

public class RaceTrackFinishTrigger : MonoBehaviour {
    void OnTriggerEnter(Collider aCarCollider) {
        var carObject = aCarCollider.gameObject;
        var carAgentComponent = carObject.GetComponent<CarAgent>();
        if (carAgentComponent != null) {
            carAgentComponent.AddReward(1.0f);
            carAgentComponent.Done();
        }
    }
}
