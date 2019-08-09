using UnityEngine;

public class SensorPropertiesComputer {
    public SensorPropertiesComputer(
            Transform aCarTransform,
            float aMaxSensorLength,
            float aOffsetY,
            float aOffsetZ) {
        CAR_TRANSFORM = aCarTransform;
        MAX_SENSOR_LENGTH = aMaxSensorLength;
        SENSOR_OFFSET_Y = aOffsetY;
        SENSOR_OFFSET_Z = aOffsetZ;
    }
    public Vector3 ComputeSensorOrigin() {
        var carPosition = CAR_TRANSFORM.position;
        Vector3 sensorOriginBeforeRotation =
                new Vector3(
                    carPosition.x,
                    carPosition.y + SENSOR_OFFSET_Y,
                    carPosition.z + SENSOR_OFFSET_Z);
        float carRotationAngle = -(CAR_TRANSFORM.eulerAngles.y);

        Vector3 sensorOriginAfterRotation =
                ComputePointRotation(
                        sensorOriginBeforeRotation,
                        carPosition,
                        carRotationAngle);
        return sensorOriginAfterRotation;
    }
    public Vector3 ComputeSensorDirection(float aCurrentSensorAngle) {
        var carPosition = CAR_TRANSFORM.position;
        Vector3 sensorOrigin =
                new Vector3(
                        carPosition.x,
                        carPosition.y + SENSOR_OFFSET_Y,
                        carPosition.z + SENSOR_OFFSET_Z);
        Vector3 sensorDirectionBeforeRotation =
                new Vector3(
                        sensorOrigin.x - MAX_SENSOR_LENGTH,
                        sensorOrigin.y,
                        sensorOrigin.z);
        Vector3 directionRotatedBySensorAngle = ComputePointRotation(
                sensorDirectionBeforeRotation,
                sensorOrigin,
                aCurrentSensorAngle);
        float carRotationAngle = CAR_TRANSFORM.eulerAngles.y;
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

    private readonly Transform CAR_TRANSFORM;
    private readonly float MAX_SENSOR_LENGTH;
    private readonly float SENSOR_OFFSET_Y;
    private readonly float SENSOR_OFFSET_Z;
}
