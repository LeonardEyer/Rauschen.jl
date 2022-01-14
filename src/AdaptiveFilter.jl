module AdaptiveFilter

    using LinearAlgebra

    function LMS(u, d, p, μ)

        # Initialize
        y = zeros(size(u, 1))
        e = zeros(size(u, 1))
        w = zeros(p)
        N = size(u, 1) - p - 1
    
        for n = 1:N
            x = reverse(u[n:n + p - 1])
            y[n] = x ⋅ w
            e[n] = d[n + p] .- y[n]

            w = w .+ μ .* x .* e[n]
        
            y[n] = x ⋅ w
        end

        return y, e, w
    end

    function NLMS(u, d, p, μ; ϵ = 0.001)

        # Initialize
        y = zeros(size(u, 1))
        e = zeros(size(u, 1))
        w = zeros(p)
        N = size(u, 1) - p - 1
    
        for n = 1:N
            x = reverse(u[n:n + p - 1])
            y[n] = x ⋅ w
            e[n] = d[n + p] .- y[n]

            normFactor = 1. / ((x ⋅ x) + ϵ)
            w = w .+ μ * normFactor .* x .* e[n]
        
            y[n] = x ⋅ w
        end

        return y, e, w
    end
end