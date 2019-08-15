using System.Collections.Generic;
using UnityEngine;

public class CarAgentOutput {
    public CarAgentOutput(float aMaxSteeringAngle, float aMotorForce) {
        MAX_STEER_ANGLE = aMaxSteeringAngle;
        MOTOR_FORCE = aMotorForce;
    }
    public void SetWheelColliders(List<WheelCollider> aColliders) {
        mWheelColliders = aColliders;
    }
    public void SetWheelTransform(List<Transform> aWheelTransform) {
        mWheelTransforms = aWheelTransform;
    }

    public void Update(float aNormalizedThrottle, float aNormalizedSteerAngle) {
        Steer(aNormalizedSteerAngle);
        Accelerate(aNormalizedThrottle);
        UpdateWheelTransform();
    }   
    
    private void Steer(float aNormalizedSteerAngle) {
        mWheelColliders[0].steerAngle = MAX_STEER_ANGLE * aNormalizedSteerAngle;
        mWheelColliders[1].steerAngle = MAX_STEER_ANGLE * aNormalizedSteerAngle;
    }
    private void Accelerate(float aNormalizedThrottle) {
        mWheelColliders[0].motorTorque = aNormalizedThrottle * MOTOR_FORCE;
        mWheelColliders[1].motorTorque = aNormalizedThrottle * MOTOR_FORCE;
    }
    private void UpdateWheelTransform() {
        for (int i = 0; i < mWheelColliders.Count && i < mWheelTransforms.Count; ++i) {
            UpdateWheelTransform(mWheelColliders[i], mWheelTransforms[i]);
        }
    }
    private void UpdateWheelTransform(
            WheelCollider aCollider,
            Transform aTransform) {
        Vector3 newWheelPosition = aTransform.position;
        Quaternion newWheelRotation = aTransform.rotation;

        aCollider.GetWorldPose(out newWheelPosition, out newWheelRotation);

        aTransform.position = newWheelPosition;
        aTransform.rotation = newWheelRotation;
    }
    
    private List<WheelCollider> mWheelColliders;
    private List<Transform> mWheelTransforms;
    private readonly float MAX_STEER_ANGLE;
    private readonly float MOTOR_FORCE;
}
