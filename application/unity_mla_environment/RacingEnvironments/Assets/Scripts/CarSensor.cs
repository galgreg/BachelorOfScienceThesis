using System;
using UnityEngine;

public class CarSensor {
    public CarSensor(float aMaxSensorLength) {
        MAX_SENSOR_LENGTH = aMaxSensorLength;
        LAYER_MASK = 1 << LayerMask.NameToLayer("RaceTrackLayer");
        mRayProperties = new Ray(Vector3.zero, Vector3.zero);
        mRaycastHit = new RaycastHit();
        mSensorRenderer = new GameObject();
        var renderingComponent = mSensorRenderer.AddComponent<LineRenderer>();
        float LINE_WIDTH = 0.01f;
        renderingComponent.startWidth = LINE_WIDTH;
        renderingComponent.endWidth = LINE_WIDTH;
        renderingComponent.material.color = SENSOR_COLOR;
        renderingComponent.positionCount = 2;
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
        Vector3 firstRendererPoint = mRayProperties.origin;
        Vector3 secondRendererPoint;

        if (mRaycastHit.collider != null) {
            secondRendererPoint = mRaycastHit.point;
        } else {
            secondRendererPoint = mRayProperties.direction;
        }
        var renderingComponent = mSensorRenderer.GetComponent<LineRenderer>();
        renderingComponent.SetPosition(0, firstRendererPoint);
        renderingComponent.SetPosition(1, secondRendererPoint);
    }

    private readonly Color SENSOR_COLOR = Color.white;

    private readonly float MAX_SENSOR_LENGTH;
    private readonly int LAYER_MASK;


    private Ray mRayProperties;
    private RaycastHit mRaycastHit;
    private GameObject mSensorRenderer;
}
