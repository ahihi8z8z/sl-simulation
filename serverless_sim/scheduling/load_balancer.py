class ShardingContainerPoolBalancer:
    """Consistent-hashing load balancer inspired by OpenWhisk."""

    def dispatch(self, invocation):
        raise NotImplementedError
