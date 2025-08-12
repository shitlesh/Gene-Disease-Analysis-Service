"""
Cost Tracking Service
Tracks token usage and calculates costs per analysis, session, and provider
Provides detailed billing analytics and cost optimization insights
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, ROUND_HALF_UP
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.session import get_async_session
from ..db.models import CostTracking, LLMProviderType, Analysis, UserSession
from ..models.llm import LLMUsageStats

logger = logging.getLogger(__name__)


class CostTrackingService:
    """Service for tracking costs and usage analytics"""
    
    # Cost per 1000 tokens (as of 2024) - update these based on current pricing
    PRICING_TABLE = {
        LLMProviderType.OPENAI: {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
        },
        LLMProviderType.ANTHROPIC: {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015}
        }
    }
    
    def __init__(self):
        self.currency = "USD"
        self.decimal_places = 6  # High precision for cost calculations
    
    async def track_analysis_cost(
        self,
        session_id: int,
        analysis_id: int,
        provider: LLMProviderType,
        model: str,
        usage_stats: LLMUsageStats,
        processing_time_ms: Optional[int] = None,
        queue_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostTracking:
        """
        Track cost for a completed analysis
        
        Args:
            session_id: User session ID
            analysis_id: Analysis ID
            provider: LLM provider
            model: Model name
            usage_stats: Token usage statistics
            processing_time_ms: Processing time in milliseconds
            queue_time_ms: Queue time in milliseconds
            metadata: Additional cost metadata
            
        Returns:
            CostTracking record
        """
        try:
            # Calculate costs
            cost_data = self._calculate_cost(provider, model, usage_stats)
            
            async with get_async_session() as session:
                # Create cost tracking record
                cost_record = CostTracking(
                    session_id=session_id,
                    analysis_id=analysis_id,
                    provider=provider,
                    model=model,
                    input_tokens=usage_stats.input_tokens,
                    output_tokens=usage_stats.output_tokens,
                    total_tokens=usage_stats.total_tokens,
                    input_cost_per_token=cost_data["input_cost_per_token"],
                    output_cost_per_token=cost_data["output_cost_per_token"],
                    total_cost=cost_data["total_cost"],
                    currency=self.currency,
                    processing_time_ms=processing_time_ms,
                    queue_time_ms=queue_time_ms,
                    cost_metadata=metadata or {}
                )
                
                session.add(cost_record)
                await session.commit()
                await session.refresh(cost_record)
                
                # Update user session totals
                await self._update_session_usage(session, session_id, usage_stats.total_tokens)
                
                logger.info(f"Tracked cost ${cost_data['total_cost']:.6f} for analysis {analysis_id}")
                return cost_record
                
        except Exception as e:
            logger.error(f"Failed to track analysis cost: {e}")
            raise
    
    def _calculate_cost(
        self, 
        provider: LLMProviderType, 
        model: str, 
        usage_stats: LLMUsageStats
    ) -> Dict[str, Any]:
        """Calculate cost based on provider pricing"""
        
        # Get pricing for the model
        pricing = self._get_model_pricing(provider, model)
        
        # Calculate costs (pricing is per 1000 tokens)
        input_cost_per_token = Decimal(str(pricing["input"])) / 1000
        output_cost_per_token = Decimal(str(pricing["output"])) / 1000
        
        input_cost = input_cost_per_token * usage_stats.input_tokens
        output_cost = output_cost_per_token * usage_stats.output_tokens
        total_cost = input_cost + output_cost
        
        return {
            "input_cost_per_token": float(input_cost_per_token.quantize(
                Decimal('0.' + '0' * self.decimal_places), rounding=ROUND_HALF_UP
            )),
            "output_cost_per_token": float(output_cost_per_token.quantize(
                Decimal('0.' + '0' * self.decimal_places), rounding=ROUND_HALF_UP
            )),
            "input_cost": float(input_cost.quantize(
                Decimal('0.' + '0' * self.decimal_places), rounding=ROUND_HALF_UP
            )),
            "output_cost": float(output_cost.quantize(
                Decimal('0.' + '0' * self.decimal_places), rounding=ROUND_HALF_UP
            )),
            "total_cost": float(total_cost.quantize(
                Decimal('0.' + '0' * self.decimal_places), rounding=ROUND_HALF_UP
            ))
        }
    
    def _get_model_pricing(self, provider: LLMProviderType, model: str) -> Dict[str, float]:
        """Get pricing for a specific model"""
        
        provider_pricing = self.PRICING_TABLE.get(provider, {})
        
        # Try exact match first
        if model in provider_pricing:
            return provider_pricing[model]
        
        # Try partial matching for model variants
        for price_model, pricing in provider_pricing.items():
            if price_model.lower() in model.lower() or model.lower() in price_model.lower():
                logger.info(f"Using pricing for {price_model} for model {model}")
                return pricing
        
        # Default to most expensive pricing if no match found
        if provider_pricing:
            most_expensive = max(provider_pricing.values(), 
                               key=lambda x: x["input"] + x["output"])
            logger.warning(f"No pricing found for {provider}/{model}, using most expensive: {most_expensive}")
            return most_expensive
        
        # Fallback pricing
        fallback = {"input": 0.01, "output": 0.03}
        logger.warning(f"No pricing available for {provider}, using fallback: {fallback}")
        return fallback
    
    async def _update_session_usage(
        self, 
        session: AsyncSession, 
        session_id: int, 
        tokens_used: int
    ) -> None:
        """Update session token usage totals"""
        try:
            from sqlmodel import update
            
            query = (
                update(UserSession)
                .where(UserSession.id == session_id)
                .values(
                    total_tokens_used=UserSession.total_tokens_used + tokens_used,
                    last_active_at=datetime.utcnow()
                )
            )
            
            await session.execute(query)
            await session.commit()
            
        except Exception as e:
            logger.error(f"Failed to update session usage: {e}")
    
    async def get_session_cost_summary(
        self, 
        session_id: int, 
        days: int = 30
    ) -> Dict[str, Any]:
        """Get cost summary for a user session"""
        try:
            async with get_async_session() as session:
                from sqlmodel import select, func
                
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Base query
                base_query = select(CostTracking).where(
                    CostTracking.session_id == session_id,
                    CostTracking.created_at >= cutoff_date
                )
                
                # Total costs and tokens
                summary_query = select(
                    func.count(CostTracking.id).label("total_analyses"),
                    func.sum(CostTracking.total_cost).label("total_cost"),
                    func.sum(CostTracking.total_tokens).label("total_tokens"),
                    func.avg(CostTracking.total_cost).label("avg_cost_per_analysis"),
                    func.avg(CostTracking.processing_time_ms).label("avg_processing_time")
                ).where(
                    CostTracking.session_id == session_id,
                    CostTracking.created_at >= cutoff_date
                )
                
                summary_result = await session.execute(summary_query)
                summary_row = summary_result.first()
                
                # Provider breakdown
                provider_query = select(
                    CostTracking.provider,
                    CostTracking.model,
                    func.count(CostTracking.id).label("count"),
                    func.sum(CostTracking.total_cost).label("cost"),
                    func.sum(CostTracking.total_tokens).label("tokens")
                ).where(
                    CostTracking.session_id == session_id,
                    CostTracking.created_at >= cutoff_date
                ).group_by(CostTracking.provider, CostTracking.model)
                
                provider_result = await session.execute(provider_query)
                provider_breakdown = []
                
                for row in provider_result.fetchall():
                    provider_breakdown.append({
                        "provider": row[0],
                        "model": row[1],
                        "analyses_count": row[2] or 0,
                        "total_cost": float(row[3] or 0),
                        "total_tokens": row[4] or 0
                    })
                
                return {
                    "session_id": session_id,
                    "period_days": days,
                    "total_analyses": summary_row[0] or 0,
                    "total_cost": float(summary_row[1] or 0),
                    "total_tokens": summary_row[2] or 0,
                    "average_cost_per_analysis": float(summary_row[3] or 0),
                    "average_processing_time_ms": float(summary_row[4] or 0),
                    "currency": self.currency,
                    "provider_breakdown": provider_breakdown
                }
                
        except Exception as e:
            logger.error(f"Failed to get session cost summary: {e}")
            return {"error": str(e)}
    
    async def get_global_cost_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get global cost analytics across all sessions"""
        try:
            async with get_async_session() as session:
                from sqlmodel import select, func
                
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Global summary
                global_query = select(
                    func.count(CostTracking.id).label("total_analyses"),
                    func.sum(CostTracking.total_cost).label("total_cost"),
                    func.sum(CostTracking.total_tokens).label("total_tokens"),
                    func.count(func.distinct(CostTracking.session_id)).label("unique_sessions"),
                    func.avg(CostTracking.total_cost).label("avg_cost_per_analysis"),
                    func.avg(CostTracking.processing_time_ms).label("avg_processing_time")
                ).where(CostTracking.created_at >= cutoff_date)
                
                global_result = await session.execute(global_query)
                global_row = global_result.first()
                
                # Top spending sessions
                top_sessions_query = select(
                    CostTracking.session_id,
                    func.sum(CostTracking.total_cost).label("total_cost"),
                    func.count(CostTracking.id).label("analysis_count")
                ).where(
                    CostTracking.created_at >= cutoff_date
                ).group_by(CostTracking.session_id).order_by(
                    func.sum(CostTracking.total_cost).desc()
                ).limit(10)
                
                top_sessions_result = await session.execute(top_sessions_query)
                top_sessions = [
                    {
                        "session_id": row[0],
                        "total_cost": float(row[1] or 0),
                        "analysis_count": row[2] or 0
                    }
                    for row in top_sessions_result.fetchall()
                ]
                
                # Cost trends by day
                daily_costs_query = select(
                    func.date(CostTracking.created_at).label("date"),
                    func.sum(CostTracking.total_cost).label("daily_cost"),
                    func.count(CostTracking.id).label("daily_analyses")
                ).where(
                    CostTracking.created_at >= cutoff_date
                ).group_by(func.date(CostTracking.created_at)).order_by("date")
                
                daily_costs_result = await session.execute(daily_costs_query)
                daily_trends = [
                    {
                        "date": str(row[0]),
                        "cost": float(row[1] or 0),
                        "analyses": row[2] or 0
                    }
                    for row in daily_costs_result.fetchall()
                ]
                
                return {
                    "period_days": days,
                    "total_analyses": global_row[0] or 0,
                    "total_cost": float(global_row[1] or 0),
                    "total_tokens": global_row[2] or 0,
                    "unique_sessions": global_row[3] or 0,
                    "average_cost_per_analysis": float(global_row[4] or 0),
                    "average_processing_time_ms": float(global_row[5] or 0),
                    "currency": self.currency,
                    "top_spending_sessions": top_sessions,
                    "daily_cost_trends": daily_trends
                }
                
        except Exception as e:
            logger.error(f"Failed to get global cost analytics: {e}")
            return {"error": str(e)}
    
    async def get_cost_optimization_insights(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Get cost optimization insights and recommendations"""
        try:
            async with get_async_session() as session:
                from sqlmodel import select, func, case
                
                # Build base filter
                base_filter = CostTracking.created_at >= datetime.utcnow() - timedelta(days=30)
                if session_id:
                    base_filter = base_filter & (CostTracking.session_id == session_id)
                
                # Model efficiency analysis
                model_efficiency_query = select(
                    CostTracking.provider,
                    CostTracking.model,
                    func.count(CostTracking.id).label("usage_count"),
                    func.avg(CostTracking.total_cost).label("avg_cost"),
                    func.avg(CostTracking.processing_time_ms).label("avg_processing_time"),
                    func.avg(CostTracking.total_tokens).label("avg_tokens"),
                    (func.avg(CostTracking.processing_time_ms) / func.avg(CostTracking.total_cost)).label("time_cost_ratio")
                ).where(base_filter).group_by(
                    CostTracking.provider, CostTracking.model
                ).order_by(func.avg(CostTracking.total_cost))
                
                efficiency_result = await session.execute(model_efficiency_query)
                model_efficiency = []
                
                for row in efficiency_result.fetchall():
                    model_efficiency.append({
                        "provider": row[0],
                        "model": row[1],
                        "usage_count": row[2] or 0,
                        "average_cost": float(row[3] or 0),
                        "average_processing_time_ms": float(row[4] or 0),
                        "average_tokens": float(row[5] or 0),
                        "efficiency_score": float(row[6] or 0)  # Lower is better
                    })
                
                # Generate recommendations
                recommendations = []
                if model_efficiency:
                    cheapest_model = min(model_efficiency, key=lambda x: x["average_cost"])
                    fastest_model = min(model_efficiency, key=lambda x: x["average_processing_time_ms"])
                    
                    recommendations.append({
                        "type": "cost_optimization",
                        "message": f"Consider using {cheapest_model['provider']}/{cheapest_model['model']} for cost optimization",
                        "potential_savings": "Variable",
                        "impact": "high"
                    })
                    
                    recommendations.append({
                        "type": "performance_optimization", 
                        "message": f"Use {fastest_model['provider']}/{fastest_model['model']} for fastest processing",
                        "performance_gain": f"{fastest_model['average_processing_time_ms']:.0f}ms average",
                        "impact": "medium"
                    })
                
                return {
                    "model_efficiency": model_efficiency,
                    "recommendations": recommendations,
                    "analysis_period_days": 30,
                    "session_id": session_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get cost optimization insights: {e}")
            return {"error": str(e)}
    
    async def update_pricing_table(self, provider: LLMProviderType, model: str, 
                                 input_cost: float, output_cost: float) -> None:
        """Update pricing table for a model (admin function)"""
        if provider not in self.PRICING_TABLE:
            self.PRICING_TABLE[provider] = {}
        
        self.PRICING_TABLE[provider][model] = {
            "input": input_cost,
            "output": output_cost
        }
        
        logger.info(f"Updated pricing for {provider}/{model}: input=${input_cost}, output=${output_cost}")
    
    def get_current_pricing(self) -> Dict[str, Any]:
        """Get current pricing table"""
        return {
            "pricing_table": self.PRICING_TABLE,
            "currency": self.currency,
            "pricing_unit": "per 1000 tokens",
            "last_updated": "2024-01-01"  # Update this when prices are refreshed
        }


# Global service instance
cost_tracking_service = CostTrackingService()