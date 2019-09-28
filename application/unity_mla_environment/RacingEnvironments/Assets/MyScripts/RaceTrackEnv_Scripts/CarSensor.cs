using UnityEngine;

public class CarSensor {
    public CarSensor(float aMaxSensorLength, uint aUnityAntibugFactor) {
        MAX_SENSOR_LENGTH = aMaxSensorLength;
        UNITY_ANTIBUG_FACTOR = aUnityAntibugFactor;
        MAX_ALLOWED_DISTANCE = MAX_SENSOR_LENGTH / UNITY_ANTIBUG_FACTOR;
        LAYER_MASK = 1 << LayerMask.NameToLayer("RaceTrackLayer");
        mRayProperties = new Ray(Vector3.zero, Vector3.zero);
        mRaycastHit = new RaycastHit();
        InitSensorRenderer();
    }
    public void SetRayProperties(Vector3 aOrigin, Vector3 aDirection) {
        mRayProperties.origin = aOrigin;
        mRayProperties.direction = aDirection;
    }
    public void SetSensorParent(Transform aNewParent) {
        mSensorRenderer.transform.parent = aNewParent;
    }
    public float GetNormalizedDistance() {
        float detectedDistance = MAX_ALLOWED_DISTANCE;
        if (WasObstacleDetected() && mRaycastHit.distance < MAX_ALLOWED_DISTANCE) {
            detectedDistance = mRaycastHit.distance;
        }
        return detectedDistance / MAX_ALLOWED_DISTANCE;
    }
    public void Render() {
        var renderingComponent = mSensorRenderer.GetComponent<LineRenderer>();
        renderingComponent.material.color = ComputeSensorColor();
        renderingComponent.SetPosition(0, mRayProperties.origin);

        if (WasObstacleDetected()) {
            renderingComponent.SetPosition(1, mRaycastHit.point);
        } else {
            renderingComponent.SetPosition(1, mRayProperties.direction);
        }
    }

    private void InitSensorRenderer() {
        mSensorRenderer = new GameObject("CarSensor");
        var renderingComponent = mSensorRenderer.AddComponent<LineRenderer>();
        float LINE_WIDTH = 0.01f;
        renderingComponent.startWidth = LINE_WIDTH;
        renderingComponent.endWidth = LINE_WIDTH;
        renderingComponent.material.color = ComputeSensorColor();
        renderingComponent.positionCount = 2;
    }

    private Color ComputeSensorColor() {
        float normalizedLength = GetNormalizedDistance();
        mCurrentSensorColor =
                Color.Lerp(MIN_LENGTH_COLOR, MAX_LENGTH_COLOR, normalizedLength);
        return mCurrentSensorColor;
    }

    private bool WasObstacleDetected() {
        return Physics.Raycast(
                mRayProperties,
                out mRaycastHit,
                MAX_SENSOR_LENGTH,
                LAYER_MASK);
    }

    private readonly Color MIN_LENGTH_COLOR = Color.red;
    private readonly Color MAX_LENGTH_COLOR = Color.green;
    private Color mCurrentSensorColor;

    private readonly float MAX_SENSOR_LENGTH;
    private readonly uint UNITY_ANTIBUG_FACTOR;
    private readonly float MAX_ALLOWED_DISTANCE;
    private readonly int LAYER_MASK;

    private Ray mRayProperties;
    private RaycastHit mRaycastHit;
    private GameObject mSensorRenderer;
}
