using System;
using UnityEngine;

public class CarSensor {
    public CarSensor() {
        LAYER_MASK = 1 << LayerMask.NameToLayer("RaceTrackLayer");
        mRayProperties = new Ray(Vector3.zero, Vector3.zero);
        mRaycastHit = new RaycastHit();
        mSensorRenderer = new LineRenderer();
    }
    public void SetRayProperties(Vector3 aOrigin, Vector3 aDirection) {
        mRayProperties.origin = aOrigin;
        mRayProperties.direction = aDirection;
    }
    
    public float GetDistance() {
        bool wasObstacleDetected = Physics.Raycast(
                mRayProperties,
                out mRaycastHit,
                MAX_SENSOR_LENGTH,
                LAYER_MASK);

        if (wasObstacleDetected) {
            return mRaycastHit.distance;
        } else {
            return MAX_SENSOR_LENGTH;
        }
    }
    public void Render() {
        // TODO
        throw new NotImplementedException();
    }

    private readonly Color MINIMUM_LENGTH_COLOR = Color.red;
    private readonly Color MAXIMUM_LENGTH_COLOR = Color.green;
    private const float MAX_SENSOR_LENGTH = 0.5f;
    private readonly int LAYER_MASK;
    
    private Ray mRayProperties;
    private RaycastHit mRaycastHit;
    private LineRenderer mSensorRenderer;
}
