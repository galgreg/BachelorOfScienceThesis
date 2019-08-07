using System;
using System.Collections.Generic;
using UnityEngine;

public class SensorsKit : MonoBehaviour {
    // Initialization code
    private void Start() {
        InitSensorList();
        // TODO -> czy trzeba tu coś jeszcze dopisać?
    }
    private void InitSensorList() {
        const float ANGLE_BETWEEN_SENSORS = FIELD_OF_VIEW / (RAYS_COUNT - 1);
        for (uint i = 0; i < RAYS_COUNT; ++i) {
            float currentSensorAngle = ANGLE_BETWEEN_SENSORS * i;
            var newSensor = CreateNewSensor(currentSensorAngle);
            mSensorList.Add(newSensor);
        }
    }
    private CarSensor CreateNewSensor(float aCurrentSensorAngle) {
        var newSensor = new CarSensor();
        Vector3 sensorOrigin = ComputeSensorOrigin();
        Vector3 sensorDirection = ComputeSensorDirection(aCurrentSensorAngle);
        newSensor.SetRayProperties(sensorOrigin, sensorDirection);
        return newSensor;
    }
    private Vector3 ComputeSensorOrigin() {
        var carPosition = mCarTransform.position;
        Vector3 sensorOriginBeforeRotation =
                new Vector3(
                    carPosition.x,
                    carPosition.y + SENSORS_ORIGIN_Y,
                    carPosition.z + SENSORS_ORIGIN_Z);
        float carRotationAngle = mCarTransform.eulerAngles.y;
        
        Vector3 sensorOriginAfterRotation =
                ComputePointRotation(
                        sensorOriginBeforeRotation,
                        carPosition,
                        carRotationAngle);
        return sensorOriginAfterRotation;
    }
    private Vector3 ComputeSensorDirection(float aCurrentSensorAngle) {
        var carPosition = mCarTransform.position;
        Vector3 sensorOrigin =
                new Vector3(
                        carPosition.x,
                        carPosition.y + SENSORS_ORIGIN_Y,
                        carPosition.z + SENSORS_ORIGIN_Z);
        Vector3 sensorDirectionBeforeRotation =
                new Vector3(
                        sensorOrigin.x - 0.5f,
                        sensorOrigin.y,
                        sensorOrigin.z);
        Vector3 directionRotatedBySensorAngle = ComputePointRotation(
                sensorDirectionBeforeRotation,
                sensorOrigin,
                aCurrentSensorAngle);
        float carRotationAngle = mCarTransform.eulerAngles.y;
        Vector3 directionRotatedByCarAngle = ComputePointRotation(
                directionRotatedBySensorAngle,
                carPosition,
                carRotationAngle);
        return directionRotatedByCarAngle;
    }
    private Vector3 ComputePointRotation(
            Vector3 pointToRotate,
            Vector3 rotationOrigin,
            float angleInDegrees) {
        float angleInRadians = angleInDegrees * Mathf.PI / 180.0f;
        float rotatedX =
                Mathf.Cos(angleInRadians) * (pointToRotate.x - rotationOrigin.x)
                - Mathf.Sin(angleInRadians) * (pointToRotate.z - rotationOrigin.z)
                + rotationOrigin.x;
        float rotatedZ =
                Mathf.Sin(angleInRadians) * (pointToRotate.x - rotationOrigin.x)
                + Mathf.Cos(angleInRadians) * (pointToRotate.z - rotationOrigin.z)
                + rotationOrigin.z;
        
        Vector3 rotatedPoint = new Vector3(rotatedX, pointToRotate.y, rotatedZ);
        return rotatedPoint;
    }

    private List<double> GetDistanceList() {
        List<double> distanceList = new List<double>();
        foreach (var sensor in mSensorList) {
            distanceList.Add(sensor.GetDistance());
        }
        return distanceList;
    }

    public Transform mCarTransform;
    public float SENSORS_ORIGIN_Y = 0.03f;
    public float SENSORS_ORIGIN_Z = 0.026f;
    private const float FIELD_OF_VIEW = 180.0f;
    private const uint RAYS_COUNT = 3;
    private List<CarSensor> mSensorList;
}
